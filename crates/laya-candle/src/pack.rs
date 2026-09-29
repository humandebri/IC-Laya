//! Canonical INT8 packs (F32 vectors and row scales). Reads one tensor at a time; never mmaps.
use crate::{expected_tensors,LayaModel,ModelConfig,Weight,int8::Int8Matrix};
use candle_core::{Device,DType,Tensor};
use ic_laya_core::{hash,BackendKind,Digest,Error,Result};
use serde::{Deserialize,Serialize};
use std::collections::BTreeMap;

#[derive(Debug,Clone,Copy,Default,PartialEq,Eq,Serialize,Deserialize)]
pub enum Storage { #[default] F32, I8Row }
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct TensorEntry {#[serde(default)] pub storage:Storage,pub name:String,pub shape:Vec<usize>,pub offset:u64,pub length:u64,pub sha256:Digest}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Manifest {
    pub format:String,pub source_repo:String,pub source_revision:String,pub test_only:bool,
    pub tokenizer_sha256:Digest,pub primitive_to_qtype:[u32;3],pub config:ModelConfig,
    pub total_bytes:u64,pub tensors:Vec<TensorEntry>,
}
impl Manifest {
    pub fn parse(raw:&[u8])->Result<Self>{
        if raw.len()>2*1024*1024{return Err(Error::TooLong);}
        let m:Self=serde_json::from_slice(raw).map_err(|e|Error::Invalid(e.to_string()))?;m.validate()?;Ok(m)
    }
    pub fn validate(&self)->Result<()> {
        if self.format!="ic-laya-int8-pack-v1" || self.source_repo.is_empty() || self.source_revision.is_empty() || self.total_bytes==0 || self.total_bytes>2*1024*1024*1024 {return Err(Error::Invalid("pack metadata".into()));}
        if !self.test_only && (self.source_revision.len()!=40 || !self.source_revision.bytes().all(|x|x.is_ascii_hexdigit())){return Err(Error::Invalid("a real pack requires an immutable source commit".into()));}
        let mut q=self.primitive_to_qtype;q.sort();if q!=[0,1,2]{return Err(Error::Invalid("qtype map".into()));}
        let expected=expected_tensors(&self.config)?;if expected.len()!=self.tensors.len(){return Err(Error::Invalid("tensor count".into()));}
        let mut pos=0u64;let mut names=std::collections::BTreeSet::new();
        for e in &self.tensors {
            if !names.insert(e.name.clone()) || expected.get(&e.name)!=Some(&e.shape) || e.offset!=pos {return Err(Error::Invalid("tensor name/shape/offset".into()));}
            let count=e.shape.iter().try_fold(1u64,|a,&b|a.checked_mul(b as u64).ok_or(Error::TooLong))?;
            let len=match e.storage {
                Storage::F32=>{
                    if e.shape.len()!=1{return Err(Error::Invalid("expected INT8 matrix storage".into()));}
                    count.checked_mul(4).ok_or(Error::TooLong)?
                },
                Storage::I8Row=>{
                    if self.format!="ic-laya-int8-pack-v1" || e.shape.len()!=2 || e.shape[1]>16384{return Err(Error::Invalid("int8 storage/shape".into()));}
                    count.checked_add(e.shape[0] as u64*4).ok_or(Error::TooLong)?
                }
            };
            if e.length!=len || len>256*1024*1024{return Err(Error::TooLong);}pos=pos.checked_add(len).ok_or(Error::TooLong)?;
        }
        if pos!=self.total_bytes{return Err(Error::Invalid("pack length".into()));}Ok(())
    }
}
/// Incremental warm-up helper shared by native and canister adapters.
pub struct Builder {pub manifest:Manifest,pub bundle:Digest,pub next:usize,tensors:BTreeMap<String,Weight>}
impl Builder {
    pub fn new(raw:&[u8])->Result<Self>{Ok(Self{manifest:Manifest::parse(raw)?,bundle:hash(raw),next:0,tensors:BTreeMap::new()})}
    pub fn next_entry(&self)->Option<&TensorEntry>{self.manifest.tensors.get(self.next)}
    pub fn push(&mut self,bytes:&[u8])->Result<()> {
        let e=self.next_entry().ok_or(Error::Transition)?.clone();
        if bytes.len() as u64!=e.length || hash(bytes)!=e.sha256{return Err(Error::Invalid("tensor integrity".into()));}
        let tensor=match e.storage {
            Storage::F32=>{
                if bytes.chunks_exact(4).any(|b|!f32::from_le_bytes([b[0],b[1],b[2],b[3]]).is_finite()){return Err(Error::Numeric);}
                Weight::F32(Tensor::from_raw_buffer(bytes,DType::F32,&e.shape,&Device::Cpu).map_err(|x|Error::ModelUnavailable(x.to_string()))?)
            },
            Storage::I8Row=>Weight::Int8(Int8Matrix::from_bytes(bytes,e.shape[0],e.shape[1]).map_err(|x|Error::Invalid(x.to_string()))?),
        };
        self.tensors.insert(e.name,tensor);self.next+=1;Ok(())
    }
    pub fn finish(self)->Result<LayaModel>{
        if self.next!=self.manifest.tensors.len(){return Err(Error::Transition);}
        let kind=if self.manifest.test_only{BackendKind::SyntheticFixture}else{BackendKind::Checkpoint};
        LayaModel::from_weights(self.manifest.config,self.bundle,kind,self.tensors)
    }
}
#[cfg(not(target_arch="wasm32"))]
pub fn load_directory(dir:&std::path::Path)->Result<LayaModel>{
    use std::io::{Read,Seek,SeekFrom};
    let raw=std::fs::read(dir.join("manifest.json")).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    let mut b=Builder::new(&raw)?;
    let mut f=std::fs::File::open(dir.join("model.bin")).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    if f.metadata().map_err(|e|Error::ModelUnavailable(e.to_string()))?.len()!=b.manifest.total_bytes{return Err(Error::Invalid("file length".into()));}
    while let Some(entry)=b.next_entry().cloned(){
        let mut bytes=vec![0u8;entry.length as usize];f.seek(SeekFrom::Start(entry.offset)).and_then(|_|f.read_exact(&mut bytes)).map_err(|e|Error::ModelUnavailable(e.to_string()))?;
        b.push(&bytes)?;
    }b.finish()
}
