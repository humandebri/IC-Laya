//! Batch-one ModernBERT + option-marker head, with W8A8 matrix weights and F32 vectors.
//! Sampled upstream parity is recorded in artifacts/laya_int8_parity.json.
//! Sampled numerical parity does not establish decision quality or calibration.
#![deny(unsafe_code)]
pub mod int8;
pub mod profile;
pub mod pack;
pub mod continuation;
pub mod quantized;
mod q8_ops;
pub mod epilogue;
use candle_core::{DType,Device,Tensor,D};
use ic_laya_core::{engine::InferenceBackend,BackendKind,Digest,Error,Result,TokenInput,MAX_TOKENS};
use serde::{Deserialize,Serialize};
use std::collections::BTreeMap;
use std::cell::RefCell;

type CResult<T> = candle_core::Result<T>;
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct ModelConfig {
    pub vocab_size:usize,pub hidden_size:usize,pub layers:usize,pub attention_heads:usize,
    pub intermediate_size:usize,pub norm_eps:f64,pub global_every:usize,pub local_attention:usize,
    pub global_rope_theta:f64,pub local_rope_theta:f64,pub first_layer_attention_norm:bool,
    pub decision_layers:usize,pub decision_heads:usize,pub decision_ff:usize,pub decision_norm_eps:f64,
    pub decision_norm_first:bool,pub decision_activation:Activation,
    pub scorer_norm_eps:f64,pub qtypes:usize,pub mask_token_id:u32,
}
#[derive(Debug,Clone,Copy,Serialize,Deserialize)]
pub enum Activation { Relu, Gelu }
impl ModelConfig {
    pub fn validate(&self)->Result<()> {
        let h=self.hidden_size;
        if h==0 || h>2048 || self.vocab_size<5 || self.vocab_size>300_000 || self.layers==0 || self.layers>64
            || self.attention_heads==0 || h%self.attention_heads!=0 || (h/self.attention_heads)%2!=0
            || self.decision_heads==0 || h%self.decision_heads!=0 || self.decision_layers==0 || self.decision_layers>4
            || self.global_every==0 || self.local_attention==0 || self.intermediate_size==0 || self.intermediate_size>16384
            || self.decision_ff==0 || self.decision_ff>16384 || self.qtypes!=3 || self.mask_token_id as usize>=self.vocab_size {
            return Err(Error::Invalid("unsupported architecture config".into()));
        }
        for v in [self.norm_eps,self.decision_norm_eps,self.scorer_norm_eps]{if !v.is_finite() || v<=0.0 || v>1.0{return Err(Error::Numeric);}}
        for v in [self.global_rope_theta,self.local_rope_theta]{if !v.is_finite() || v<=1.0 || v>1e12{return Err(Error::Numeric);}}Ok(())
    }
}
/// Runtime storage: F32 is restricted to one-dimensional bias/norm vectors.
#[derive(Clone)]
pub enum Weight { F32(Tensor), Int8(int8::Int8Matrix) }
impl Weight {
    fn dims(&self)->Vec<usize>{match self{Self::F32(t)=>t.dims().to_vec(),Self::Int8(t)=>t.dims().to_vec()}}
}
#[derive(Clone)]
pub struct Linear { weight:int8::Int8Matrix,bias:Option<Tensor> }
impl Linear {
    pub fn forward(&self,x:&Tensor)->CResult<Tensor>{let y=self.weight.forward(x)?;match &self.bias{Some(b)=>y.broadcast_add(b),None=>Ok(y)}}
}

#[derive(Clone)]
pub struct Norm { weight:Tensor,bias:Option<Tensor>,eps:f64 }
impl Norm {
    pub fn new(weight:Tensor,bias:Option<Tensor>,eps:f64)->Self{Self{weight,bias,eps}}
    pub fn forward(&self,x:&Tensor)->CResult<Tensor>{
        profile::measure("norm",[0,0,0],||{
        let mean=x.mean_keepdim(D::Minus1)?;
        let centered=x.broadcast_sub(&mean)?;
        let var=centered.sqr()?.mean_keepdim(D::Minus1)?;
        let y=centered.broadcast_div(&(var+self.eps)?.sqrt()?)?.broadcast_mul(&self.weight)?;
        match &self.bias{Some(b)=>y.broadcast_add(b),None=>Ok(y)}
        })
    }
}
#[derive(Clone)]
pub struct Attention { qkv:Linear,out:Linear,heads:usize }
fn softmax_last(x:&Tensor)->CResult<Tensor>{let e=x.broadcast_sub(&x.max_keepdim(D::Minus1)?)?.exp()?;e.broadcast_div(&e.sum_keepdim(D::Minus1)?)}
thread_local! {
    // The encoder uses the same two theta values at every layer. Keep only a few
    // CPU tables so different model configurations cannot grow this cache forever.
    static ROPE_TABLES: RefCell<Vec<((u64,usize,usize),Tensor,Tensor)>> = const { RefCell::new(Vec::new()) };
}
fn rope(x:&Tensor,theta:f64)->CResult<Tensor>{
    let (_,t,d)=x.dims3()?;let half=d/2;
    let tables=if matches!(x.device(),Device::Cpu){
        ROPE_TABLES.with(|cache| -> CResult<(Tensor,Tensor)> {
            let key=(theta.to_bits(),t,d);
            let mut cache=cache.borrow_mut();
            if let Some((_,cos,sin))=cache.iter().find(|(k,_,_)|*k==key){return Ok((cos.clone(),sin.clone()));}
            let mut cos=Vec::with_capacity(t*half);let mut sin=Vec::with_capacity(t*half);
            for pos in 0..t {for i in 0..half {let a=pos as f64/theta.powf((2*i) as f64/d as f64);cos.push(a.cos() as f32);sin.push(a.sin() as f32);}}
            let cos=Tensor::from_vec(cos,(1,t,half),x.device())?;let sin=Tensor::from_vec(sin,(1,t,half),x.device())?;
            if cache.len()==4 {cache.remove(0);}
            cache.push((key,cos.clone(),sin.clone()));
            Ok((cos,sin))
        })?
    }else{
        let mut cos=Vec::with_capacity(t*half);let mut sin=Vec::with_capacity(t*half);
        for pos in 0..t {for i in 0..half {let a=pos as f64/theta.powf((2*i) as f64/d as f64);cos.push(a.cos() as f32);sin.push(a.sin() as f32);}}
        (Tensor::from_vec(cos,(1,t,half),x.device())?,Tensor::from_vec(sin,(1,t,half),x.device())?)
    };
    let (cos,sin)=tables;
    let a=x.narrow(2,0,half)?;let b=x.narrow(2,half,half)?;
    let left=(a.broadcast_mul(&cos)?-b.broadcast_mul(&sin)?)?;
    let right=(b.broadcast_mul(&cos)?+a.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&left,&right],2)
}
impl Attention {
    pub fn new(qkv:Linear,out:Linear,heads:usize)->Self{Self{qkv,out,heads}}
    pub fn forward(&self,x:&Tensor,rotary:Option<f64>,max_distance:Option<usize>)->CResult<Tensor>{
        let (t,h)=x.dims2()?;let d=h/self.heads;let y=self.qkv.forward(x)?;
        let mut q=y.narrow(1,0,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let mut k=y.narrow(1,h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let v=y.narrow(1,2*h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        if let Some(theta)=rotary {(q,k)=profile::measure("attention.rope",[t,h,self.heads],||Ok::<_,candle_core::Error>((rope(&q,theta)?,rope(&k,theta)?)))?;}
        let mut scores=profile::measure("attention.qk",[t,h,self.heads],||q.matmul(&k.transpose(1,2)?)?*(1.0/(d as f64).sqrt()))?;
        if let Some(distance)=max_distance {
            let mask:Vec<f32>=(0..t).flat_map(|i|(0..t).map(move|j|if i.abs_diff(j)>distance{f32::NEG_INFINITY}else{0.0})).collect();
            scores=scores.broadcast_add(&Tensor::from_vec(mask,(1,t,t),x.device())?)?;
        }
        let probs=profile::measure("attention.softmax",[t,h,self.heads],||softmax_last(&scores))?;
        let merged=profile::measure("attention.av",[t,h,self.heads],||probs.matmul(&v)?.transpose(0,1)?.contiguous()?.reshape((t,h)))?;
        self.out.forward(&merged)
    }
    // On the final decision layer, only marker rows are used by the scorer.
    // Keys and values still cover the whole sequence, preserving attention.
    fn forward_markers(&self,x:&Tensor,markers:&Tensor)->CResult<Tensor>{
        let (t,h)=x.dims2()?;let d=h/self.heads;let m=markers.dim(0)?;
        let y=self.qkv.forward(x)?;
        let q=y.narrow(1,0,h)?.contiguous()?.index_select(markers,0)?.reshape((m,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let k=y.narrow(1,h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let v=y.narrow(1,2*h,h)?.reshape((t,self.heads,d))?.transpose(0,1)?.contiguous()?;
        let scores=profile::measure("attention.qk",[m,h,self.heads],||q.matmul(&k.transpose(1,2)?)?*(1.0/(d as f64).sqrt()))?;
        let probs=profile::measure("attention.softmax",[m,h,self.heads],||softmax_last(&scores))?;
        let merged=profile::measure("attention.av",[m,h,self.heads],||probs.matmul(&v)?.transpose(0,1)?.contiguous()?.reshape((m,h)))?;
        self.out.forward(&merged)
    }
}
#[derive(Clone)]
pub struct EncoderLayer { attention_norm:Option<Norm>,attention:Attention,mlp_norm:Norm,wi:Linear,wo:Linear,theta:f64,distance:Option<usize> }
impl EncoderLayer {
    pub fn new(attention_norm:Option<Norm>,attention:Attention,mlp_norm:Norm,wi:Linear,wo:Linear,theta:f64,distance:Option<usize>)->Self{
        Self{attention_norm,attention,mlp_norm,wi,wo,theta,distance}
    }
    pub fn forward(&self,x:&Tensor)->CResult<Tensor>{self.forward_marked(x,&mut |_|{})}
    /// Same computation, with a marker after each sub-phase.
    ///
    /// The split matters because the two halves have different fixes: the attention
    /// projections and the MLP are dense matmuls (quantisation territory), whereas
    /// softmax and the gated activation are element-wise and would need different
    /// treatment.
    fn forward_marked(&self,x:&Tensor,mark:&mut dyn FnMut(&'static str))->CResult<Tensor>{
        let normalized=match &self.attention_norm{Some(n)=>n.forward(x)?,None=>x.clone()};
        mark("layer.attn_norm");
        let attended=self.attention.forward(&normalized,Some(self.theta),self.distance)?;
        mark("layer.attn");
        let x=(x+attended)?;
        mark("layer.attn_resid");
        let y=self.wi.forward(&self.mlp_norm.forward(&x)?)?;let half=y.dim(1)?/2;
        mark("layer.mlp_up");
        let gate=profile::measure("mlp.gelu_gate",[y.dim(0)?,half,0],||(y.narrow(1,0,half)?.gelu_erf()?*y.narrow(1,half,half)?))?;
        mark("layer.mlp_act");
        let out=(&x+self.wo.forward(&gate)?)?;
        mark("layer.mlp_down");
        Ok(out)
    }
}
#[derive(Clone)]
struct DecisionLayer { norm1:Norm,norm2:Norm,attention:Attention,linear1:Linear,linear2:Linear,norm_first:bool,activation:Activation }
impl DecisionLayer {
    fn ff(&self,x:&Tensor)->CResult<Tensor>{let x=self.linear1.forward(x)?;self.linear2.forward(&match self.activation{Activation::Relu=>x.relu()?,Activation::Gelu=>x.gelu_erf()?})}
    fn forward(&self,x:&Tensor)->CResult<Tensor>{
        if self.norm_first {
            let x=(x+self.attention.forward(&self.norm1.forward(x)?,None,None)?)?;
            &x+self.ff(&self.norm2.forward(&x)?)?
        }else{
            let x=self.norm1.forward(&(x+self.attention.forward(x,None,None)?)?)?;
            self.norm2.forward(&(&x+self.ff(&x)?)?)
        }
    }
    fn forward_markers(&self,x:&Tensor,markers:&Tensor)->CResult<Tensor>{
        let selected=x.index_select(markers,0)?;
        if self.norm_first {
            let x=(&selected+self.attention.forward_markers(&self.norm1.forward(x)?,markers)?)?;
            &x+self.ff(&self.norm2.forward(&x)?)?
        }else{
            let x=self.norm1.forward(&(&selected+self.attention.forward_markers(x,markers)?)?)?;
            self.norm2.forward(&(&x+self.ff(&x)?)?)
        }
    }
}

pub fn expected_tensors(c:&ModelConfig)->Result<BTreeMap<String,Vec<usize>>>{
    c.validate()?;let mut m=BTreeMap::new();let h=c.hidden_size;
    m.insert("embeddings.weight".into(),vec![c.vocab_size,h]);m.insert("embeddings.norm.weight".into(),vec![h]);m.insert("final_norm.weight".into(),vec![h]);
    for i in 0..c.layers {
        let p=format!("encoder.{i}");
        if i>0 || c.first_layer_attention_norm{m.insert(format!("{p}.attn_norm.weight"),vec![h]);}
        for (name,shape) in [("qkv.weight",vec![3*h,h]),("out.weight",vec![h,h]),("mlp_norm.weight",vec![h]),("wi.weight",vec![2*c.intermediate_size,h]),("wo.weight",vec![h,c.intermediate_size])] {m.insert(format!("{p}.{name}"),shape);}
    }
    m.insert("qtype.weight".into(),vec![c.qtypes,h]);
    for i in 0..c.decision_layers {
        let p=format!("decision.{i}");
        for (name,shape) in [("qkv.weight",vec![3*h,h]),("qkv.bias",vec![3*h]),("out.weight",vec![h,h]),("out.bias",vec![h]),("norm1.weight",vec![h]),("norm1.bias",vec![h]),("norm2.weight",vec![h]),("norm2.bias",vec![h]),("linear1.weight",vec![c.decision_ff,h]),("linear1.bias",vec![c.decision_ff]),("linear2.weight",vec![h,c.decision_ff]),("linear2.bias",vec![h])] {m.insert(format!("{p}.{name}"),shape);}
    }
    for (name,shape) in [("norm.weight",vec![h]),("norm.bias",vec![h]),("dense.weight",vec![h,h]),("dense.bias",vec![h]),("out.weight",vec![1,h]),("out.bias",vec![1])] {m.insert(format!("scorer.{name}"),shape);}
    Ok(m)
}

/// In-memory continuation. A step runs one encoder/head layer (or final/scorer phase).
/// The pack bundle is checked on every step, so model replacement cannot mix weights.
#[derive(Clone)]
pub struct InferenceSession { bundle:Digest,input:TokenInput,hidden:Tensor,next:usize }
impl InferenceSession { pub fn completed_steps(&self)->usize{self.next} }

pub struct LayaModel {
    pub config:ModelConfig,pub bundle:Digest,pub backend_kind:BackendKind,
    embedding:int8::Int8Matrix,embedding_norm:Norm,layers:Vec<EncoderLayer>,final_norm:Norm,qtype:int8::Int8Matrix,
    decision:Vec<DecisionLayer>,scorer_norm:Norm,scorer_dense:Linear,scorer_out:Linear,
}
fn weight(m:&BTreeMap<String,Weight>,name:&str)->Result<Weight>{m.get(name).cloned().ok_or_else(||Error::Invalid(format!("missing tensor: {name}")))}
fn tensor(m:&BTreeMap<String,Weight>,name:&str)->Result<Tensor>{match weight(m,name)?{Weight::F32(t)=>Ok(t),Weight::Int8(_)=>Err(Error::Invalid(format!("expected F32: {name}")))}}
fn matrix(m:&BTreeMap<String,Weight>,name:&str)->Result<int8::Int8Matrix>{match weight(m,name)?{Weight::Int8(t)=>Ok(t),Weight::F32(_)=>Err(Error::Invalid(format!("expected INT8 matrix: {name}")))}}
fn linear(m:&BTreeMap<String,Weight>,p:&str,bias:bool)->Result<Linear>{Ok(Linear{weight:matrix(m,&format!("{p}.weight"))?,bias:if bias{Some(tensor(m,&format!("{p}.bias"))?)}else{None}})}
fn norm(m:&BTreeMap<String,Weight>,p:&str,bias:bool,eps:f64)->Result<Norm>{Ok(Norm{weight:tensor(m,&format!("{p}.weight"))?,bias:if bias{Some(tensor(m,&format!("{p}.bias"))?)}else{None},eps})}
impl LayaModel {
    pub fn from_weights(c:ModelConfig,bundle:Digest,kind:BackendKind,m:BTreeMap<String,Weight>)->Result<Self>{
        let expected=expected_tensors(&c)?;
        if expected.len()!=m.len(){return Err(Error::Invalid("unexpected tensor set".into()));}
        for (name,shape) in &expected {let t=m.get(name).ok_or_else(||Error::Invalid(format!("missing {name}")))?;if t.dims()!=*shape || matches!(t,Weight::F32(v) if v.dtype()!=DType::F32 || shape.len()!=1){return Err(Error::Invalid(format!("shape/dtype: {name}")));}}
        let mut layers=Vec::new();
        for i in 0..c.layers {let p=format!("encoder.{i}");let local=i%c.global_every!=0;
            layers.push(EncoderLayer{attention_norm:if i>0||c.first_layer_attention_norm{Some(norm(&m,&format!("{p}.attn_norm"),false,c.norm_eps)?)}else{None},
                attention:Attention{qkv:linear(&m,&format!("{p}.qkv"),false)?,out:linear(&m,&format!("{p}.out"),false)?,heads:c.attention_heads},mlp_norm:norm(&m,&format!("{p}.mlp_norm"),false,c.norm_eps)?,
                wi:linear(&m,&format!("{p}.wi"),false)?,wo:linear(&m,&format!("{p}.wo"),false)?,theta:if local{c.local_rope_theta}else{c.global_rope_theta},distance:if local{Some(c.local_attention/2)}else{None}});
        }
        let mut decision=Vec::new();
        for i in 0..c.decision_layers {let p=format!("decision.{i}");decision.push(DecisionLayer{norm1:norm(&m,&format!("{p}.norm1"),true,c.decision_norm_eps)?,norm2:norm(&m,&format!("{p}.norm2"),true,c.decision_norm_eps)?,
            attention:Attention{qkv:linear(&m,&format!("{p}.qkv"),true)?,out:linear(&m,&format!("{p}.out"),true)?,heads:c.decision_heads},linear1:linear(&m,&format!("{p}.linear1"),true)?,linear2:linear(&m,&format!("{p}.linear2"),true)?,norm_first:c.decision_norm_first,activation:c.decision_activation});}
        Ok(Self{embedding:matrix(&m,"embeddings.weight")?,embedding_norm:norm(&m,"embeddings.norm",false,c.norm_eps)?,final_norm:norm(&m,"final_norm",false,c.norm_eps)?,qtype:matrix(&m,"qtype.weight")?,
            scorer_norm:norm(&m,"scorer.norm",true,c.scorer_norm_eps)?,scorer_dense:linear(&m,"scorer.dense",true)?,scorer_out:linear(&m,"scorer.out",true)?,config:c,bundle,backend_kind:kind,layers,decision})
    }
    fn validate_input(&self,input:&TokenInput)->Result<()>{
        if input.input_ids.is_empty() || input.input_ids.len()>MAX_TOKENS || !(2..=7).contains(&input.markers.len()) || input.qtype_id as usize>=self.config.qtypes
            || input.input_ids.iter().any(|&v|v as usize>=self.config.vocab_size){return Err(Error::Invalid("token input".into()));}
        let mut prev=None;
        for &p in &input.markers {if input.input_ids.get(p as usize)!=Some(&self.config.mask_token_id) || prev.is_some_and(|v|p<=v){return Err(Error::Invalid("option markers".into()));}prev=Some(p);}
        Ok(())
    }
    pub fn inference_steps(&self)->usize{self.layers.len()+self.decision.len()+2}
    pub fn begin_inference(&self,input:TokenInput)->Result<InferenceSession>{
        self.validate_input(&input)?;
        let hidden=self.embedding.gather(&input.input_ids).and_then(|x|self.embedding_norm.forward(&x)).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        Ok(InferenceSession{bundle:self.bundle,input,hidden,next:0})
    }
    pub fn step_inference(&self,session:&mut InferenceSession)->Result<Option<Vec<f32>>>{
        if session.bundle!=self.bundle{return Err(Error::BindingMismatch);}
        let n=session.next;
        if n>=self.inference_steps(){return Err(Error::Transition);}
        let run=||->CResult<Tensor>{
            if n<self.layers.len(){self.layers[n].forward(&session.hidden)}
            else if n==self.layers.len(){
                self.final_norm.forward(&session.hidden)?.broadcast_add(&self.qtype.gather(&[session.input.qtype_id])?)
            }else if n<self.inference_steps()-1{
                let i=n-self.layers.len()-1;
                if i+1==self.decision.len(){
                    let markers=Tensor::from_vec(session.input.markers.clone(),session.input.markers.len(),&Device::Cpu)?;
                    self.decision[i].forward_markers(&session.hidden,&markers)
                }else{self.decision[i].forward(&session.hidden)}
            }else{
                self.scorer_out.forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&session.hidden)?)?.gelu_erf()?)?.squeeze(1)
            }
        };
        let hidden=run().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        let logits=if n==self.inference_steps()-1{
            let values=hidden.to_vec1::<f32>().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
            if values.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}
            Some(values)
        }else{None};
        session.hidden=hidden;session.next+=1;
        Ok(logits)
    }
    fn forward_tensor(&self,input:&TokenInput)->CResult<Tensor>{
        let dev=&Device::Cpu;
        let mut x=self.embedding_norm.forward(&self.embedding.gather(&input.input_ids)?)?;
        for layer in &self.layers{x=layer.forward(&x)?;}
        x=self.final_norm.forward(&x)?;
        let qt=self.qtype.gather(&[input.qtype_id])?;x=x.broadcast_add(&qt)?;
        let markers=Tensor::from_vec(input.markers.clone(),input.markers.len(),dev)?;
        for (i,layer) in self.decision.iter().enumerate(){
            x=if i+1==self.decision.len(){layer.forward_markers(&x,&markers)?}else{layer.forward(&x)?};
        }
        self.scorer_out.forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&x)?)?.gelu_erf()?)?.squeeze(1)
    }

    /// `forward_tensor` with a callback at each phase boundary.
    ///
    /// Measurement only: `InferenceBackend::infer` does not use this, so the
    /// production path carries no instrumentation. The callback is a plain
    /// function pointer and this crate does not depend on `ic-cdk`, so it works
    /// on native (where the counter is a stub returning 0) and inside a canister
    /// (where it reads `instruction_counter`) without a second implementation.
    fn forward_profiled(&self,input:&TokenInput,mark:&mut dyn FnMut(&'static str),detailed:bool)->CResult<Tensor>{
        let dev=&Device::Cpu;
        let mut x=self.embedding_norm.forward(&self.embedding.gather(&input.input_ids)?)?;
        mark("embedding");
        if detailed {
            for layer in &self.layers{x=layer.forward_marked(&x,mark)?;}
        } else {
            for layer in &self.layers{x=layer.forward(&x)?;}
        }
        mark("encoder");
        x=self.final_norm.forward(&x)?;
        let qt=self.qtype.gather(&[input.qtype_id])?;x=x.broadcast_add(&qt)?;
        mark("final_norm");
        let markers=Tensor::from_vec(input.markers.clone(),input.markers.len(),dev)?;
        for (i,layer) in self.decision.iter().enumerate(){
            x=if i+1==self.decision.len(){layer.forward_markers(&x,&markers)?}else{layer.forward(&x)?};
        }
        mark("decision");
        let out=self.scorer_out.forward(&self.scorer_dense.forward(&self.scorer_norm.forward(&x)?)?.gelu_erf()?)?.squeeze(1);
        mark("scorer");
        out
    }

    /// Per-phase instruction cost for one question.
    ///
    /// Returns one entry per phase in execution order. `instructions` is the
    /// delta since the previous boundary, so the entries sum to the total for
    /// this call (the `decode` phase covers `.to_vec1` and the finite check).
    pub fn infer_profiled(&mut self,input:&TokenInput,counter:&dyn Fn()->u64)->Result<Vec<PhaseCost>>{
        self.infer_profiled_detailed(input,counter,false)
    }

    /// As `infer_profiled`, but also marks the sub-phases of every encoder layer.
    /// The per-layer entries repeat, so callers should aggregate by name.
    pub fn infer_profiled_detailed(&mut self,input:&TokenInput,counter:&dyn Fn()->u64,detailed:bool)->Result<Vec<PhaseCost>>{
        self.validate_input(input)?;
        let mut costs:Vec<PhaseCost>=Vec::new();
        let mut last=counter();
        {
            let mut record=|name:&'static str|{let now=counter();costs.push(PhaseCost{name,instructions:now.saturating_sub(last)});last=now;};
            let tensor=self.forward_profiled(input,&mut record,detailed).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
            let values=tensor.to_vec1::<f32>().map_err(|e|Error::ModelUnavailable(e.to_string()))?;
            record("decode");
            if values.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}
        }
        Ok(costs)
    }
}

/// One phase of a profiled inference. Measurement only; never part of a Receipt.
#[derive(Debug,Clone,PartialEq,Eq,Serialize,Deserialize)]
pub struct PhaseCost { pub name:&'static str, pub instructions:u64 }
impl InferenceBackend for LayaModel {
    fn bundle_id(&self)->Digest{self.bundle}
    fn kind(&self)->BackendKind{self.backend_kind}
    fn infer(&mut self,input:&TokenInput)->Result<Vec<f32>>{
        self.validate_input(input)?;
        let v=self.forward_tensor(input).and_then(|t|t.to_vec1::<f32>()).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        if v.iter().any(|x|!x.is_finite()){return Err(Error::Numeric);}Ok(v)
    }
}

#[cfg(test)]
mod optimized_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn final_decision_marker_rows_match_full_layer() {
        for folder in ["tiny-int8-prenorm", "tiny-int8-postnorm"] {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(folder);
            let model = pack::load_directory(&dir).unwrap();
            let cases: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("cases.json")).unwrap()).unwrap();
            for case in cases.as_array().unwrap() {
                let input: TokenInput = serde_json::from_value(case["input"].clone()).unwrap();
                let mut hidden = model.embedding_norm.forward(&model.embedding.gather(&input.input_ids).unwrap()).unwrap();
                for layer in &model.layers { hidden = layer.forward(&hidden).unwrap(); }
                hidden = model.final_norm.forward(&hidden).unwrap().broadcast_add(&model.qtype.gather(&[input.qtype_id]).unwrap()).unwrap();
                for layer in &model.decision[..model.decision.len() - 1] { hidden = layer.forward(&hidden).unwrap(); }
                let markers = Tensor::from_vec(input.markers.clone(), input.markers.len(), &Device::Cpu).unwrap();
                let last = model.decision.last().unwrap();
                let full = last.forward(&hidden).unwrap().index_select(&markers, 0).unwrap().to_vec2::<f32>().unwrap();
                let selected = last.forward_markers(&hidden, &markers).unwrap().to_vec2::<f32>().unwrap();
                assert_eq!(selected, full, "{folder}");
            }
        }
    }
}
