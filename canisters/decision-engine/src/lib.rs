//! Default build starts unavailable. Synthetic fixture mode is an explicit admin action.
use candid::{CandidType,Principal};
use ic_laya_core::{engine::EngineState,schema,*,demo::{FixtureBackend,FixtureTokenizer}};
use serde::{Serialize,Deserialize};
use std::cell::RefCell;

#[cfg(feature="candle")]
mod query_transport;

/// Supplies the getrandom 0.3 backend that candle-core and tokenizers need on
/// wasm32-unknown-unknown. Must stay in the canister (wasm link root) and
/// unconditional, because `.cargo/config.toml` selects that backend for the whole
/// wasm32 target. Compiles to nothing on native targets.
#[cfg(target_arch = "wasm32")]
mod getrandom_ic;

#[derive(Clone,Serialize,Deserialize)]
struct Upload {manifest:Vec<u8>,tokenizer_length:u64,model_length:u64,received:u64,special:SpecialTokens}
#[derive(Clone,Serialize,Deserialize)]
struct Persistent {owner:Principal,engine:EngineState,fixture_mode:bool,upload:Option<Upload>}
thread_local!{static STATE:RefCell<Option<Persistent>>=const{RefCell::new(None)};}
#[cfg(feature="candle")]
thread_local!{
    static TOKENIZER:RefCell<Option<hf_tokenizer::HfTokenizer>>=const{RefCell::new(None)};
    static BUILDER:RefCell<Option<laya_candle::pack::Builder>>=const{RefCell::new(None)};
    static MODEL:RefCell<Option<laya_candle::LayaModel>>=const{RefCell::new(None)};
    static INFERENCE:RefCell<Option<TokenJob>>=const{RefCell::new(None)};
}
fn read<R>(f:impl FnOnce(&Persistent)->R)->R{STATE.with(|x|f(x.borrow().as_ref().expect("initialized")))}
fn mutate<R>(f:impl FnOnce(&mut Persistent)->R)->R{STATE.with(|x|{let mut state=x.borrow_mut();let state=state.as_mut().expect("initialized");let result=f(state);canister_common::persist_or_trap(state);result})}
fn check_owner(s:&Persistent,caller:Principal)->Result<()>{if caller==s.owner{Ok(())}else{Err(Error::Unauthorized)}}
fn owner()->Result<()>{read(|s|check_owner(s,ic_cdk::api::msg_caller()))}
/// Recheck persisted schemas against the currently active pack, including after replacement.
#[cfg(feature="candle")]
fn check_schema_mapping(s:&Persistent,id:&Digest)->Result<()> {
    if s.fixture_mode{return Ok(());}
    let schema=s.engine.schemas.get(id).ok_or(Error::NotFound)?;
    let upload=s.upload.as_ref().ok_or(Error::Transition)?;
    if hash(&upload.manifest)!=s.engine.active_model{return Err(Error::BindingMismatch);}
    let manifest=laya_candle::pack::Manifest::parse(&upload.manifest)?;
    if schema.qtype_id!=manifest.primitive_to_qtype[schema.schema.primitive.tag() as usize]{return Err(Error::BindingMismatch);}
    Ok(())
}
#[ic_cdk::init]
fn init(owner:Principal){
    if owner==Principal::anonymous() || owner==Principal::management_canister(){ic_cdk::trap("invalid owner");}
    let s=Persistent{owner,engine:EngineState::new([0;32]),fixture_mode:false,upload:None};canister_common::persist_or_trap(&s);STATE.with(|x|*x.borrow_mut()=Some(s));
}
#[ic_cdk::pre_upgrade]
fn pre_upgrade(){read(canister_common::persist_or_trap);}
#[ic_cdk::post_upgrade]
fn post_upgrade(){let s:Persistent=canister_common::restore().unwrap_or_else(|e|ic_cdk::trap(&e.to_string()));STATE.with(|x|*x.borrow_mut()=Some(s));}
#[derive(CandidType,Serialize,Deserialize)]
pub struct EngineInfo {pub model:Digest,pub fixture:bool,pub schemas:u64,pub cached:u64,pub live_dispatch_supported:bool}
#[ic_cdk::query]
fn info()->EngineInfo{read(|s|EngineInfo{model:s.engine.active_model,fixture:s.fixture_mode,schemas:s.engine.schemas.len() as u64,cached:s.engine.cache.len() as u64,live_dispatch_supported:false})}
#[ic_cdk::update]
fn allow_caller(caller:Principal,per_minute:u32)->Result<()>{owner()?;mutate(|s|s.engine.allow_caller(caller,per_minute))}
#[ic_cdk::update]
fn enable_synthetic_fixture()->Result<Digest>{
    owner()?;let bundle=ic_laya_core::demo::fixture_bundle();
    mutate(|s|{if s.upload.is_some(){return Err(Error::Transition);}s.fixture_mode=true;s.engine.active_model=bundle;Ok(bundle)})
}
#[ic_cdk::update]
fn register_schema(schema:Schema,qtype_id:u32)->Result<CompiledSchema>{
    owner()?;
    if !read(|s|s.fixture_mode) {
        #[cfg(feature="candle")]
        {
            let upload=read(|s|s.upload.clone()).ok_or(Error::Transition)?;
            let manifest=laya_candle::pack::Manifest::parse(&upload.manifest)?;
            if manifest.primitive_to_qtype[schema.primitive.tag() as usize]!=qtype_id {
                return Err(Error::BindingMismatch);
            }
        }
    }
    let compiled=if read(|s|s.fixture_mode){schema::compile(schema,&FixtureTokenizer,qtype_id)?}else{
        #[cfg(feature="candle")]
        {TOKENIZER.with(|t|schema::compile(schema,t.borrow().as_ref().ok_or_else(||Error::ModelUnavailable("warm-up required".into()))?,qtype_id))?}
        #[cfg(not(feature="candle"))]
        {return Err(Error::ModelUnavailable("build decision-engine with --features candle".into()));}
    };
    mutate(|s|s.engine.register(compiled.clone()))?;Ok(compiled)
}
#[ic_cdk::update]
fn register_calibration(c:Calibration)->Result<()>{owner()?;mutate(|s|s.engine.register_calibration(c,ic_cdk::api::time()))}
/// MEASUREMENT ONLY. Reports where inference instructions are spent, phase by phase.
///
/// This exists because the acceptance targets are instruction budgets and nothing
/// reported a breakdown: `evaluate` returns one total for the whole call. It uses the
/// owner authorization and request validation but **bypasses the cache**, since
/// a cached result would report nothing.
///
/// It cannot move funds: it does not touch the executor, does not create a Receipt,
/// and is not reachable from any dispatch path. Gated on the candle feature because
/// the fixture backend has no real phases to report.
/// Wire form of `laya_candle::PhaseCost`. Kept local because deriving CandidType on
/// the crate type would make `laya-candle` depend on candid for a type that only
/// exists for measurement.
#[derive(Clone,CandidType,Serialize,Deserialize)]
pub struct PhaseCostDto { pub name:String, pub instructions:u64 }

#[cfg(feature="candle")]
#[ic_cdk::update]
fn measure_phases(req:DecisionRequest)->Result<Vec<PhaseCostDto>>{
    owner()?;
    let now=ic_cdk::api::time();
    if req.state.is_empty() || req.state.len()>MAX_STATE_BYTES{return Err(Error::TooLong);}
    if req.expires_at_ns<=now || req.expires_at_ns-now>600_000_000_000{return Err(Error::Expired);}
    // Validate against registered state exactly as evaluate would, so the measured
    // phases describe a request that would actually be accepted.
    let schema=read(|s|s.engine.schemas.get(&req.schema_hash).cloned()).ok_or(Error::NotFound)?;
    read(|s|{
        check_schema_mapping(s,&req.schema_hash)?;
        if req.model!=s.engine.active_model{return Err(Error::BindingMismatch);}
        if let Some(id)=req.calibration{
            let c=s.engine.calibrations.get(&id).ok_or(Error::Uncalibrated)?;
            c.validate(now)?;
            if c.model!=req.model || c.schema!=req.schema_hash || c.tokenizer!=schema.tokenizer_hash{return Err(Error::BindingMismatch);}
            if req.expires_at_ns>c.expires_at_ns{return Err(Error::Uncalibrated);}
        }Ok(())
    })?;
    TOKENIZER.with(|t|MODEL.with(|m|{
        let t=t.borrow();let mut m=m.borrow_mut();
        let (t,m)=match (t.as_ref(),m.as_mut()){(Some(t),Some(m))=>(t,m),_=>return Err(Error::ModelUnavailable("checkpoint not warmed".into()))};
        let input=crate::schema::render(&schema,t,&req.state)?;
        m.infer_profiled_detailed(&input,&||ic_cdk::api::instruction_counter(),true)
            .map(|costs|costs.into_iter().map(|c|PhaseCostDto{name:c.name.to_string(),instructions:c.instructions}).collect())
    }))
}
#[ic_cdk::update]
fn evaluate(req:DecisionRequest)->Result<Receipt>{
    let caller=ic_cdk::api::msg_caller();let now=ic_cdk::api::time();
    // Reject outsiders before serializing the durable engine snapshot.
    if !read(|s|s.engine.callers.contains_key(&caller)){return Err(Error::Unauthorized);}
    if req.state.is_empty() || req.state.len()>MAX_STATE_BYTES{return Err(Error::TooLong);}
    // A reserve for administration/recovery; this is not a cycles-per-decision benchmark.
    if ic_cdk::api::canister_cycle_balance()<100_000_000_000 {return Err(Error::Capacity);}
    #[cfg(feature="candle")]
    read(|s|check_schema_mapping(s,&req.schema_hash))?;
    let start=ic_cdk::api::instruction_counter();
    mutate(|s|{
        let cached=s.engine.cache.contains_key(&(caller,req.evaluation_id));
        let key=(caller,req.evaluation_id);
        let mut result=if s.fixture_mode {
            s.engine.evaluate(caller,req,now,&FixtureTokenizer,&mut FixtureBackend::default())
        }else{
            #[cfg(feature="candle")]
            {TOKENIZER.with(|t|MODEL.with(|m|{
                let t=t.borrow();let mut m=m.borrow_mut();
                match (t.as_ref(),m.as_mut()){
                    (Some(t),Some(m))=>s.engine.evaluate(caller,req,now,t,m),
                    _=>Err(Error::ModelUnavailable("checkpoint not warmed".into())),
                }
            }))}
            #[cfg(not(feature="candle"))]
            {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
        };
        if !cached {
            if let Ok(r)=&mut result{r.measured_instructions=ic_cdk::api::instruction_counter().saturating_sub(start);}
            if let Some(entry)=s.engine.cache.get_mut(&key){entry.result=result.clone();}
        }result
    })
}
#[ic_cdk::update]
fn begin_upload(manifest:Vec<u8>,tokenizer_length:u64,special:SpecialTokens)->Result<Digest>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let m=laya_candle::pack::Manifest::parse(&manifest)?;
        if m.config.mask_token_id!=special.mask{return Err(Error::BindingMismatch);}
        if tokenizer_length==0 || tokenizer_length>32*1024*1024{return Err(Error::TooLong);}
        let bundle=hash(&manifest);
        // No second active model is retained during replacement.
        INFERENCE.with(|x|*x.borrow_mut()=None);MODEL.with(|x|*x.borrow_mut()=None);BUILDER.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=None);
        mutate(|s|{s.fixture_mode=false;s.upload=Some(Upload{manifest,tokenizer_length,model_length:m.total_bytes,received:0,special});Ok(bundle)})
    }
    #[cfg(not(feature="candle"))]
    {let _=(manifest,tokenizer_length,special);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[ic_cdk::update]
fn upload_chunk(offset:u64,bytes:Vec<u8>)->Result<u64>{
    owner()?;if bytes.is_empty() || bytes.len()>1024*1024{return Err(Error::TooLong);}
    mutate(|s|{
        let u=s.upload.as_mut().ok_or(Error::Transition)?;
        let end=offset.checked_add(bytes.len() as u64).ok_or(Error::TooLong)?;
        if end>u.model_length+u.tokenizer_length{return Err(Error::TooLong);}
        if offset<u.received {
            if end>u.received{return Err(Error::IdConflict);}let mut old=vec![0;bytes.len()];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+offset,&mut old);
            return if old==bytes{Ok(u.received)}else{Err(Error::IdConflict)};
        }
        if offset!=u.received{return Err(Error::NonceGap);}
        canister_common::grow_through(canister_common::BLOB_BASE+end)?;
        ic_cdk::stable::stable_write(canister_common::BLOB_BASE+offset,&bytes);u.received=end;Ok(end)
    })
}
#[ic_cdk::update]
fn start_warmup()->Result<u64>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let u=read(|s|s.upload.clone()).ok_or(Error::Transition)?;
        if u.received!=u.model_length+u.tokenizer_length{return Err(Error::Transition);}
        let builder=laya_candle::pack::Builder::new(&u.manifest)?;
        let mut bytes=vec![0;u.tokenizer_length as usize];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+u.model_length,&mut bytes);
        if hash(&bytes)!=builder.manifest.tokenizer_sha256{return Err(Error::BindingMismatch);}
        let tok=hf_tokenizer::HfTokenizer::from_bytes(&bytes,u.special)?;
        let count=builder.manifest.tensors.len() as u64;
        INFERENCE.with(|x|*x.borrow_mut()=None);MODEL.with(|x|*x.borrow_mut()=None);TOKENIZER.with(|x|*x.borrow_mut()=Some(tok));BUILDER.with(|x|*x.borrow_mut()=Some(builder));Ok(count)
    }
    #[cfg(not(feature="candle"))]
    {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[ic_cdk::update]
fn warmup_next()->Result<bool>{warmup_next_inner()}
fn warmup_next_inner()->Result<bool>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let done=BUILDER.with(|b|{
            let mut b=b.borrow_mut();let builder=b.as_mut().ok_or(Error::Transition)?;
            if let Some(e)=builder.next_entry().cloned(){
                let mut bytes=vec![0;e.length as usize];ic_cdk::stable::stable_read(canister_common::BLOB_BASE+e.offset,&mut bytes);builder.push(&bytes)?;
            }
            Ok::<bool,Error>(builder.next_entry().is_none())
        })?;
        if done {
            let builder=BUILDER.with(|b|b.borrow_mut().take()).ok_or(Error::Transition)?;let bundle=builder.bundle;let model=builder.finish()?;
            MODEL.with(|m|*m.borrow_mut()=Some(model));mutate(|s|s.engine.active_model=bundle);
        }Ok(done)
    }
    #[cfg(not(feature="candle"))]
    {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[derive(CandidType,Deserialize)]
pub struct WarmupProfile {pub done:bool,pub instructions:u64}
/// Owner-only measurement; each call loads one tensor just like warmup_next.
#[ic_cdk::update]
fn warmup_next_profile()->Result<WarmupProfile>{
    let start=ic_cdk::api::instruction_counter();
    let done=warmup_next_inner()?;
    Ok(WarmupProfile{done,instructions:ic_cdk::api::instruction_counter().saturating_sub(start)})
}
#[cfg(feature="candle")]
struct TokenJob { id:Digest,session:laya_candle::InferenceSession,instructions:u64,total:u32,last:Option<InferenceProgress>,last_request:Option<(u32,u32)>,start_request:Option<(Digest,Digest,InferenceProgress)> }
#[cfg(feature="candle")]
impl TokenJob {
    fn replay_start(&self,request_id:Digest,fingerprint:Digest)->Result<Option<InferenceProgress>>{
        if let Some((id,old,result))=&self.start_request{
            if *id==request_id{return if *old==fingerprint{Ok(Some(result.clone()))}else{Err(Error::IdConflict)};}
        }
        Ok(None)
    }
    fn advance(&mut self,model:&laya_candle::LayaModel,job:Digest,expected_step:u32,max_steps:u32,counter:fn()->u64)->Result<InferenceProgress>{
        if !(1..=16).contains(&max_steps){return Err(Error::Invalid("max_steps must be 1..=16".into()));}
        if self.id!=job{return Err(Error::BindingMismatch);}
        if self.last_request==Some((expected_step,max_steps)){return self.last.clone().ok_or(Error::Transition);}
        if self.session.completed_steps()!=expected_step as usize{return Err(Error::BindingMismatch);}
        if expected_step>=self.total{return Err(Error::Transition);}
        let start=counter();
        // Tensor clones share immutable storage. Commit only after every requested layer succeeds.
        let mut session=self.session.clone();
        let mut logits=None;
        for _ in 0..max_steps.min(self.total-expected_step){logits=model.step_inference(&mut session)?;}
        let instructions=counter().saturating_sub(start);
        let cumulative_instructions=self.instructions.saturating_add(instructions);
        let result=InferenceProgress{job,completed:session.completed_steps() as u32,total:self.total,logits,instructions,cumulative_instructions};
        self.session=session;self.instructions=cumulative_instructions;
        self.last=Some(result.clone());self.last_request=Some((expected_step,max_steps));
        Ok(result)
    }
}
#[derive(Debug,Clone,PartialEq,CandidType,Deserialize)]
pub struct InferenceProgress { pub job:Digest,pub completed:u32,pub total:u32,pub logits:Option<Vec<f32>>,pub instructions:u64,pub cumulative_instructions:u64 }
/// Start one owner-only, in-memory continuation; a new request replaces the old one.
/// Upgrade drops it. The uploaded model remains in stable memory.
#[ic_cdk::update]
fn start_token_inference(input:TokenInput)->Result<InferenceProgress>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let start=ic_cdk::api::instruction_counter();
        let (session,total,bundle)=MODEL.with(|m|{
            let m=m.borrow();let m=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
            Ok::<_,Error>((m.begin_inference(input.clone())?,m.inference_steps() as u32,m.bundle))
        })?;
        let now=ic_cdk::api::time();
        let old=INFERENCE.with(|j|j.borrow().as_ref().map(|j|j.id));
        let id=hash(&candid::encode_args((bundle,now,old,input)).map_err(|e|Error::Invalid(e.to_string()))?);
        let instructions=ic_cdk::api::instruction_counter().saturating_sub(start);
        INFERENCE.with(|j|*j.borrow_mut()=Some(TokenJob{id,session,instructions,total,last:None,last_request:None,start_request:None}));
        Ok(InferenceProgress{job:id,completed:0,total,logits:None,instructions,cumulative_instructions:instructions})
    }
    #[cfg(not(feature="candle"))]
    {let _=input;Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
/// Start and run the first batch in one update. request_id makes retries of the current job idempotent.
#[ic_cdk::update]
fn start_token_inference_batch(input:TokenInput,request_id:Digest,max_steps:u32)->Result<InferenceProgress>{
    owner()?;
    #[cfg(feature="candle")]
    {
        if !(1..=16).contains(&max_steps){return Err(Error::Invalid("max_steps must be 1..=16".into()));}
        let bundle=MODEL.with(|m|m.borrow().as_ref().map(|m|m.bundle)).ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        let fingerprint=hash(&candid::encode_args((&input,max_steps,bundle)).map_err(|e|Error::Invalid(e.to_string()))?);
        if let Some(replay)=INFERENCE.with(|j|j.borrow().as_ref().map(|j|j.replay_start(request_id,fingerprint)).transpose().map(Option::flatten))?{return Ok(replay);}
        let start=ic_cdk::api::instruction_counter();
        let old=INFERENCE.with(|j|j.borrow().as_ref().map(|j|j.id));
        let id=hash(&candid::encode_args((bundle,ic_cdk::api::time(),old,request_id,fingerprint)).map_err(|e|Error::Invalid(e.to_string()))?);
        let (mut job,mut result)=MODEL.with(|m|{
            let m=m.borrow();let m=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
            let session=m.begin_inference(input)?;
            let mut job=TokenJob{id,session,instructions:ic_cdk::api::instruction_counter().saturating_sub(start),
                total:m.inference_steps() as u32,last:None,last_request:None,start_request:None};
            let result=job.advance(m,id,0,max_steps,ic_cdk::api::instruction_counter)?;
            Ok::<_,Error>((job,result))
        })?;
        result.instructions=result.cumulative_instructions;
        job.last=Some(result.clone());job.start_request=Some((request_id,fingerprint,result.clone()));
        INFERENCE.with(|j|*j.borrow_mut()=Some(job));
        Ok(result)
    }
    #[cfg(not(feature="candle"))]
    {let _=(input,request_id,max_steps);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
/// Exact retries return the cached result; a stale request cannot advance twice.
#[ic_cdk::update]
fn step_token_inference(job:Digest,expected_step:u32)->Result<InferenceProgress>{
    owner()?;
    advance_token_job(job,expected_step,1)
}
/// Up to sixteen consecutive phases per update. Retry with the same start and limit.
/// For larger/custom packs reduce max_steps if the update instruction limit is reached.
#[ic_cdk::update]
fn step_token_inference_batch(job:Digest,expected_step:u32,max_steps:u32)->Result<InferenceProgress>{
    owner()?;
    advance_token_job(job,expected_step,max_steps)
}
fn advance_token_job(job:Digest,expected_step:u32,max_steps:u32)->Result<InferenceProgress>{
    #[cfg(feature="candle")]
    {INFERENCE.with(|j|MODEL.with(|m|{
        let mut j=j.borrow_mut();let j=j.as_mut().ok_or(Error::NotFound)?;
        let m=m.borrow();let m=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        j.advance(m,job,expected_step,max_steps,ic_cdk::api::instruction_counter)
    }))}
    #[cfg(not(feature="candle"))]
    {let _=(job,expected_step,max_steps);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[ic_cdk::query]
fn token_inference_status()->Result<InferenceProgress>{
    owner()?;
    #[cfg(feature="candle")]
    {INFERENCE.with(|j|{
        let j=j.borrow();let j=j.as_ref().ok_or(Error::NotFound)?;
        Ok(j.last.clone().unwrap_or(InferenceProgress{job:j.id,completed:0,total:j.total,logits:None,instructions:j.instructions,cumulative_instructions:j.instructions}))
    })}
    #[cfg(not(feature="candle"))]
    {Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[derive(CandidType,Deserialize)]
pub struct ProfileCost { pub name:String,pub shape:Vec<u64>,pub instructions:u64 }
#[derive(CandidType,Deserialize)]
pub struct ProfiledStep { pub progress:InferenceProgress,pub costs:Vec<ProfileCost> }
/// Owner-only bounded synthetic kernel measurements; no uploaded model is required.
#[derive(CandidType)]
pub struct KernelBenchmark { pub instructions:u64,pub checksum:Digest,pub costs:Vec<ProfileCost> }
#[cfg(feature="candle")]
#[ic_cdk::update]
fn benchmark_int8_kernel(tokens:u32,rows:u32,cols:u32)->Result<KernelBenchmark>{
    owner()?;
    if !(1..=128).contains(&tokens) || ![1024,3072,5248].contains(&rows) || ![1024,2624].contains(&cols){return Err(Error::Invalid("benchmark shape".into()));}
    let (instructions,checksum,costs)=laya_candle::int8::benchmark(tokens as usize,rows as usize,cols as usize,ic_cdk::api::instruction_counter)
        .map_err(|e|Error::ModelUnavailable(e.to_string()))?;
    Ok(KernelBenchmark{instructions,checksum,costs:costs.into_iter().map(|c|ProfileCost{name:c.name,shape:c.shape.into_iter().map(|v|v as u64).collect(),instructions:c.instructions}).collect()})
}
/// Diagnostic breakdown for the three dominant 128-token matrix shapes.
#[derive(CandidType)]
pub struct KernelComponents { pub total:u64,pub integer_dots:u64,pub f32_writeback:u64,pub checksum:Digest }
#[cfg(feature="candle")]
#[ic_cdk::update]
fn benchmark_int8_components(rows:u32,cols:u32)->Result<KernelComponents>{
    owner()?;
    let (total,integer_dots,f32_writeback,checksum)=laya_candle::int8::benchmark_components(rows as usize,cols as usize,ic_cdk::api::instruction_counter)
        .map_err(|e|Error::Invalid(e.to_string()))?;
    Ok(KernelComponents{total,integer_dots,f32_writeback,checksum})
}
#[ic_cdk::update]
fn profile_token_step(job:Digest,expected_step:u32)->Result<ProfiledStep>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let (progress,costs)=laya_candle::profile::capture(ic_cdk::api::instruction_counter,||step_token_inference(job,expected_step));
        Ok(ProfiledStep{progress:progress?,costs:costs.into_iter().map(|c|ProfileCost{name:c.name,shape:c.shape.into_iter().map(|v|v as u64).collect(),instructions:c.instructions}).collect()})
    }
    #[cfg(not(feature="candle"))]
    {let _=(job,expected_step);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

/// Owner-only token-level inference, for model parity and cost validation.
/// Does not issue a Receipt or authorize an executor action.
#[derive(candid::CandidType, serde::Deserialize)]
pub struct TokenInference { pub logits:Vec<f32>, pub instructions:u64 }
/// Measured on the fixed Laya W8A8 pack with 2..=7 markers and all three qtypes.
/// The 16-token worst case stays below 5B; every measured 17-token case exceeds it.
const QUERY_MAX_TOKENS: usize = 16;
#[cfg(feature="candle")]
const QUERY_BENCHMARKED_BUNDLE: Digest = [
    0xbb,0x70,0xb3,0xf0,0xf2,0x80,0x6b,0xef,0x5d,0x4b,0x67,0x0f,0x44,0xbb,0x60,0x68,
    0x92,0x06,0x7f,0xc0,0xeb,0xd9,0x28,0xbd,0x68,0x2b,0x98,0xeb,0xdb,0x2d,0xc0,0x92,
];
fn infer_tokens_inner(input:TokenInput)->Result<TokenInference>{
    #[cfg(feature="candle")]
    {
        use ic_laya_core::engine::InferenceBackend;
        let start=ic_cdk::api::instruction_counter();
        let logits=MODEL.with(|m|m.borrow_mut().as_mut().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?.infer(&input))?;
        Ok(TokenInference{logits,instructions:ic_cdk::api::instruction_counter().saturating_sub(start)})
    }
    #[cfg(not(feature="candle"))]
    {let _=input;Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[ic_cdk::update]
fn infer_tokens(input:TokenInput)->Result<TokenInference>{
    owner()?;
    infer_tokens_inner(input)
}
/// Owner-only short-input query for the benchmarked pack. Returns raw,
/// uncertified logits and no Receipt. Rejects 17+ tokens before inference.
#[ic_cdk::query]
fn infer_tokens_query(input:TokenInput)->Result<TokenInference>{
    owner()?;
    if input.input_ids.len()>QUERY_MAX_TOKENS{return Err(Error::TooLong);}
    #[cfg(feature="candle")]
    MODEL.with(|cell|{
        let model=cell.borrow();
        let model=model.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        if model.bundle!=QUERY_BENCHMARKED_BUNDLE{return Err(Error::BindingMismatch);}
        Ok::<_,Error>(())
    })?;
    infer_tokens_inner(input)
}

/// Independent queries return the entire continuation to the caller. No TokenJob is created.
/// Instructions cover the handler, including continuation conversion, but exclude CDK decoding/encoding.
#[derive(Debug,Clone,PartialEq,CandidType,Deserialize)]
pub enum QueryInferenceProgress {
    Continue { state:Vec<u8>, completed:u32, total:u32, instructions:u64 },
    Done { logits:Vec<f32>, completed:u32, total:u32, instructions:u64 },
}

fn decode_query_begin(bytes:Vec<u8>)->TokenInput {
    owner().unwrap_or_else(|_|ic_cdk::trap("Unauthorized"));
    if bytes.len()>4096 {ic_cdk::trap("query input too large");}
    let mut config=candid::de::DecoderConfig::new();
    config.set_decoding_quota(20_000).set_skipping_quota(1000);
    candid::decode_args_with_config::<(TokenInput,)>(&bytes,&config)
        .unwrap_or_else(|_|ic_cdk::trap("invalid query input")).0
}
fn decode_query_begin_batch(bytes:Vec<u8>)->(TokenInput,u32) {
    owner().unwrap_or_else(|_|ic_cdk::trap("Unauthorized"));
    if bytes.len()>4096 {ic_cdk::trap("query input too large");}
    let mut config=candid::de::DecoderConfig::new();
    config.set_decoding_quota(20_000).set_skipping_quota(1000);
    candid::decode_args_with_config(&bytes,&config)
        .unwrap_or_else(|_|ic_cdk::trap("invalid batched query input"))
}
fn decode_query_continue(bytes:Vec<u8>)->(Vec<u8>,u32) {
    owner().unwrap_or_else(|_|ic_cdk::trap("Unauthorized"));
    // The largest supported architecture needs just over 1 MiB; allow a bounded Candid envelope.
    if bytes.len()>1_100_000 {ic_cdk::trap("query continuation too large");}
    let mut config=candid::de::DecoderConfig::new();
    config.set_decoding_quota(4_000_000).set_skipping_quota(1000);
    candid::decode_args_with_config(&bytes,&config)
        .unwrap_or_else(|_|ic_cdk::trap("invalid query continuation"))
}

#[cfg(feature="candle")]
fn check_query_pack(model:&laya_candle::LayaModel)->Result<()> {
    if model.bundle!=QUERY_BENCHMARKED_BUNDLE {return Err(Error::BindingMismatch);}
    Ok(())
}

#[cfg(feature="candle")]
fn query_begin(model:&laya_candle::LayaModel,input:TokenInput,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    let start=counter();
    let session=model.begin_inference(input)?;
    let state=model.export_inference(&session)?;
    Ok(QueryInferenceProgress::Continue{state,completed:0,total:model.inference_steps() as u32,instructions:counter().saturating_sub(start)})
}

#[cfg(feature="candle")]
fn query_continue(model:&laya_candle::LayaModel,state:&[u8],max_steps:u32,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    if !(1..=16).contains(&max_steps) {return Err(Error::Invalid("query max_steps must be 1..=16".into()));}
    let start=counter();
    let mut session=model.import_inference(state)?;
    let total=model.inference_steps() as u32;
    for _ in 0..max_steps.min(total-session.completed_steps() as u32) {
        if let Some(logits)=model.step_inference(&mut session)? {
            return Ok(QueryInferenceProgress::Done{logits,completed:total,total,instructions:counter().saturating_sub(start)});
        }
    }
    let completed=session.completed_steps() as u32;
    let state=model.export_inference(&session)?;
    Ok(QueryInferenceProgress::Continue{state,completed,total,instructions:counter().saturating_sub(start)})
}

/// Fixed-pack, owner-only raw inference; the continuation is not authenticated.
// Custom CDK decoding otherwise exports a blob-only Candid signature. Register
// the actual typed contract explicitly while keeping the bounded decoder.
#[ic_cdk::query(hidden=true,decode_with="decode_query_begin")]
#[candid::candid_method(query)]
fn begin_token_inference_query(input:TokenInput)->Result<QueryInferenceProgress> {
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;
        query_begin(model,input,ic_cdk::api::instruction_counter)
    })}
    #[cfg(not(feature="candle"))]
    {let _=input;Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[ic_cdk::query(hidden=true,decode_with="decode_query_continue")]
#[candid::candid_method(query)]
fn continue_token_inference_query(state:Vec<u8>,max_steps:u32)->Result<QueryInferenceProgress> {
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;
        query_continue(model,&state,max_steps,ic_cdk::api::instruction_counter)
    })}
    #[cfg(not(feature="candle"))]
    {let _=(state,max_steps);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
/// Experimental activation-INT8 engine; numerical parity is not assumed.
#[cfg(feature="candle")]
fn query_begin_int8(model:&laya_candle::LayaModel,input:TokenInput,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    let start=counter();
    let session=model.begin_quantized(input)?;
    let state=model.export_quantized(&session)?;
    Ok(QueryInferenceProgress::Continue{state,completed:0,total:model.inference_steps() as u32,instructions:counter().saturating_sub(start)})
}

#[cfg(feature="candle")]
fn query_continue_int8(model:&laya_candle::LayaModel,state:&[u8],max_steps:u32,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    if !(1..=16).contains(&max_steps){return Err(Error::Invalid("query max_steps must be 1..=16".into()));}
    let start=counter();let mut session=model.import_quantized(state)?;let total=model.inference_steps() as u32;
    for _ in 0..max_steps.min(total-session.completed_steps() as u32){
        if let Some(logits)=model.step_quantized(&mut session)?{
            return Ok(QueryInferenceProgress::Done{logits,completed:total,total,instructions:counter().saturating_sub(start)});
        }
    }
    let completed=session.completed_steps() as u32;let state=model.export_quantized(&session)?;
    Ok(QueryInferenceProgress::Continue{state,completed,total,instructions:counter().saturating_sub(start)})
}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin")]
#[candid::candid_method(query)]
fn begin_token_inference_int8_query(input:TokenInput)->Result<QueryInferenceProgress>{
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;query_begin_int8(model,input,ic_cdk::api::instruction_counter)
    })}
    #[cfg(not(feature="candle"))]
    {let _=input;Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[ic_cdk::query(hidden=true,decode_with="decode_query_continue")]
#[candid::candid_method(query)]
fn continue_token_inference_int8_query(state:Vec<u8>,max_steps:u32)->Result<QueryInferenceProgress>{
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;query_continue_int8(model,&state,max_steps,ic_cdk::api::instruction_counter)
    })}
    #[cfg(not(feature="candle"))]
    {let _=(state,max_steps);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[derive(CandidType,Deserialize)]
pub struct ProfiledQuery { pub progress:QueryInferenceProgress,pub costs:Vec<ProfileCost> }

#[cfg(feature="candle")]
fn query_begin_batch(model:&laya_candle::LayaModel,input:TokenInput,max_steps:u32,int8:bool,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    if !(1..=16).contains(&max_steps){return Err(Error::Invalid("query max_steps must be 1..=16".into()));}
    let start=counter();let total=model.inference_steps() as u32;
    let (state,completed)=if int8 {
        let mut session=model.begin_quantized(input)?;
        for _ in 0..max_steps.min(total){
            if let Some(logits)=model.step_quantized(&mut session)?{
                return Ok(QueryInferenceProgress::Done{logits,completed:total,total,instructions:counter().saturating_sub(start)});
            }
        }
        (model.export_quantized(&session)?,session.completed_steps() as u32)
    } else {
        let mut session=model.begin_inference(input)?;
        for _ in 0..max_steps.min(total){
            if let Some(logits)=model.step_inference(&mut session)?{
                return Ok(QueryInferenceProgress::Done{logits,completed:total,total,instructions:counter().saturating_sub(start)});
            }
        }
        (model.export_inference(&session)?,session.completed_steps() as u32)
    };
    Ok(QueryInferenceProgress::Continue{state,completed,total,instructions:counter().saturating_sub(start)})
}

#[cfg(feature="candle")]
fn transported_query(model:&laya_candle::LayaModel,input:Option<TokenInput>,wire:&[u8],max_steps:u32,int8:bool,counter:impl Fn()->u64)->Result<QueryInferenceProgress> {
    let start=counter();
    let progress=if let Some(input)=input {
        if max_steps>0 {query_begin_batch(model,input,max_steps,int8,||0)?}
        else if int8 {query_begin_int8(model,input,||0)?} else {query_begin(model,input,||0)?}
    } else {
        let raw=query_transport::unpack(wire)?;
        if int8 {query_continue_int8(model,&raw,max_steps,||0)?} else {query_continue(model,&raw,max_steps,||0)?}
    };
    Ok(match progress {
        QueryInferenceProgress::Continue{state,completed,total,..}=>{
            let state=query_transport::pack(state)?;
            QueryInferenceProgress::Continue{state,completed,total,instructions:counter().saturating_sub(start)}
        },
        QueryInferenceProgress::Done{logits,completed,total,..}=>QueryInferenceProgress::Done{logits,completed,total,instructions:counter().saturating_sub(start)},
    })
}

fn transported_call(input:Option<TokenInput>,wire:&[u8],max_steps:u32,int8:bool)->Result<QueryInferenceProgress> {
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;
        transported_query(model,input,wire,max_steps,int8,ic_cdk::api::instruction_counter)
    })}
    #[cfg(not(feature="candle"))]
    {let _=(input,wire,max_steps,int8);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin")]
#[candid::candid_method(query)]
fn begin_token_inference_compressed_query(input:TokenInput)->Result<QueryInferenceProgress>{transported_call(Some(input),&[],0,false)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_continue")]
#[candid::candid_method(query)]
fn continue_token_inference_compressed_query(state:Vec<u8>,max_steps:u32)->Result<QueryInferenceProgress>{transported_call(None,&state,max_steps,false)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin")]
#[candid::candid_method(query)]
fn begin_token_inference_int8_compressed_query(input:TokenInput)->Result<QueryInferenceProgress>{transported_call(Some(input),&[],0,true)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_continue")]
#[candid::candid_method(query)]
fn continue_token_inference_int8_compressed_query(state:Vec<u8>,max_steps:u32)->Result<QueryInferenceProgress>{transported_call(None,&state,max_steps,true)}

fn begin_batch_call(input:TokenInput,max_steps:u32,int8:bool,compressed:bool)->Result<QueryInferenceProgress> {
    owner()?;
    if !(1..=16).contains(&max_steps){return Err(Error::Invalid("query max_steps must be 1..=16".into()));}
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;
        if compressed {transported_query(model,Some(input),&[],max_steps,int8,ic_cdk::api::instruction_counter)}
        else {query_begin_batch(model,input,max_steps,int8,ic_cdk::api::instruction_counter)}
    })}
    #[cfg(not(feature="candle"))]
    {let _=(input,max_steps,int8,compressed);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin_batch")]
#[candid::candid_method(query)]
fn begin_token_inference_batch_query(input:TokenInput,max_steps:u32)->Result<QueryInferenceProgress>{begin_batch_call(input,max_steps,false,false)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin_batch")]
#[candid::candid_method(query)]
fn begin_token_inference_int8_batch_query(input:TokenInput,max_steps:u32)->Result<QueryInferenceProgress>{begin_batch_call(input,max_steps,true,false)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin_batch")]
#[candid::candid_method(query)]
fn begin_token_inference_batch_compressed_query(input:TokenInput,max_steps:u32)->Result<QueryInferenceProgress>{begin_batch_call(input,max_steps,false,true)}

#[ic_cdk::query(hidden=true,decode_with="decode_query_begin_batch")]
#[candid::candid_method(query)]
fn begin_token_inference_int8_batch_compressed_query(input:TokenInput,max_steps:u32)->Result<QueryInferenceProgress>{begin_batch_call(input,max_steps,true,true)}

/// Owner-only bounded query diagnostic; performs no persistent job writes.
#[ic_cdk::query(hidden=true,decode_with="decode_query_continue")]
#[candid::candid_method(query)]
fn profile_token_inference_int8_query(state:Vec<u8>,max_steps:u32)->Result<ProfiledQuery>{
    owner()?;
    #[cfg(feature="candle")]
    {MODEL.with(|m|{
        let m=m.borrow();let model=m.as_ref().ok_or_else(||Error::ModelUnavailable("model is cold".into()))?;
        check_query_pack(model)?;
        let (progress,costs)=laya_candle::profile::capture(ic_cdk::api::instruction_counter,||query_continue_int8(model,&state,max_steps,ic_cdk::api::instruction_counter));
        Ok(ProfiledQuery{progress:progress?,costs:costs.into_iter().map(|c|ProfileCost{name:c.name,shape:c.shape.into_iter().map(|v|v as u64).collect(),instructions:c.instructions}).collect()})
    })}
    #[cfg(not(feature="candle"))]
    {let _=(state,max_steps);Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
#[derive(CandidType)]
pub struct EpilogueVariant { pub name:String,pub instructions:u64,pub checksum:Digest,pub clamped:u64,pub rmse:f64,pub max_abs_error:f32,pub different_from_fixed_float:u64 }
#[derive(CandidType)]
pub struct EpilogueBenchmark { pub prepare_instructions:u64,pub variants:Vec<EpilogueVariant> }
fn decode_epilogue_benchmark(bytes:Vec<u8>)->Vec<u8>{
    owner().unwrap_or_else(|_|ic_cdk::trap("Unauthorized"));
    if bytes.len()>2_800_000 {ic_cdk::trap("epilogue benchmark input too large");}
    let mut config=candid::de::DecoderConfig::new();config.set_decoding_quota(9_000_000).set_skipping_quota(1000);
    candid::decode_args_with_config::<(Vec<u8>,)>(&bytes,&config).unwrap_or_else(|_|ic_cdk::trap("invalid epilogue benchmark input")).0
}
/// Bounded diagnostic on supplied real accumulation data. Calibration is not
/// installed into MODEL, nor used by Receipt inference.
#[ic_cdk::query(decode_with="decode_epilogue_benchmark")]
fn benchmark_q8_epilogue(input:Vec<u8>)->Result<EpilogueBenchmark>{
    owner()?;
    #[cfg(feature="candle")]
    {
        let input=laya_candle::epilogue::Input::decode(&input).map_err(|e|Error::Invalid(e.to_string()))?;
        let report=laya_candle::epilogue::benchmark(&input,ic_cdk::api::instruction_counter).map_err(|_|Error::Numeric)?;
        Ok(EpilogueBenchmark{prepare_instructions:report.prepare_instructions,variants:report.variants.into_iter().map(|v|EpilogueVariant{name:v.name,instructions:v.instructions,checksum:v.checksum,clamped:v.clamped,rmse:v.rmse,max_abs_error:v.max_abs_error,different_from_fixed_float:v.different_from_fixed_float}).collect()})
    }
    #[cfg(not(feature="candle"))]
    {let _=input;Err(Error::ModelUnavailable("Candle feature disabled".into()))}
}
ic_cdk::export_candid!();
pub fn candid_interface()->String{__export_service()}

#[cfg(all(test,feature="candle"))]
mod client_query_tests;

#[cfg(all(test,feature="candle"))]
mod tests {
    use super::*;
    use ic_laya_core::{demo::{actor,schemas},schema::TextTokenizer};

    fn state()->Persistent {
        let manifest=include_bytes!("../../../fixtures/tiny-int8-prenorm/manifest.json").to_vec();
        Persistent{owner:actor(1),engine:EngineState::new(hash(&manifest)),fixture_mode:false,
            upload:Some(Upload{manifest,tokenizer_length:1,model_length:1,received:2,
                special:FixtureTokenizer.special_tokens()})}
    }

    #[test]
    fn batch_retries_bounds_and_mixed_single_steps() {
        use ic_laya_core::engine::InferenceBackend;
        fn counter()->u64{0}
        let dir=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tiny-int8-prenorm");
        let mut model=laya_candle::pack::load_directory(&dir).unwrap();
        let input:TokenInput=serde_json::from_slice(&std::fs::read(dir.join("input.json")).unwrap()).unwrap();
        let expected=model.infer(&input).unwrap();
        let id=[7;32];
        let mut job=TokenJob{id,session:model.begin_inference(input).unwrap(),instructions:0,
            total:model.inference_steps() as u32,last:None,last_request:None,start_request:None};
        for limit in [0,17,u32::MAX]{assert!(matches!(job.advance(&model,id,0,limit,counter),Err(Error::Invalid(_))));}
        assert_eq!(job.session.completed_steps(),0);
        assert_eq!(job.advance(&model,[8;32],0,2,counter),Err(Error::BindingMismatch));
        let first=job.advance(&model,id,0,2,counter).unwrap();
        assert_eq!(first.completed,2);
        job.start_request=Some(([9;32],[10;32],first.clone()));
        assert_eq!(job.replay_start([9;32],[10;32]),Ok(Some(first.clone())));
        assert_eq!(job.replay_start([9;32],[11;32]),Err(Error::IdConflict));
        assert_eq!(job.replay_start([8;32],[10;32]),Ok(None));
        assert_eq!(job.advance(&model,id,0,2,counter).unwrap(),first);
        assert_eq!(job.advance(&model,id,0,3,counter),Err(Error::BindingMismatch));
        assert_eq!(job.advance(&model,id,1,1,counter),Err(Error::BindingMismatch));
        // A failed call must not discard progress or the cached reply.
        model.bundle[0]^=1;
        assert_eq!(job.advance(&model,id,2,2,counter),Err(Error::BindingMismatch));
        assert_eq!(job.session.completed_steps(),2);
        assert_eq!(job.last.as_ref(),Some(&first));
        model.bundle[0]^=1;
        let single=job.advance(&model,id,2,1,counter).unwrap();
        assert_eq!(job.advance(&model,id,2,1,counter).unwrap(),single);
        let last=job.advance(&model,id,3,16,counter).unwrap();
        assert_eq!(last.completed,job.total);
        assert_eq!(last.logits,Some(expected));
        assert_eq!(job.replay_start([9;32],[10;32]),Ok(Some(first)));
        assert_eq!(job.advance(&model,id,3,16,counter).unwrap(),last);
        assert_eq!(job.advance(&model,id,0,2,counter),Err(Error::BindingMismatch));
        assert_eq!(job.advance(&model,id,job.total,16,counter),Err(Error::Transition));
    }

    #[test]
    fn allowed_inference_caller_is_not_a_profiling_owner() {
        let mut s=state();
        s.engine.allow_caller(actor(2),1).unwrap();
        assert_eq!(check_owner(&s,actor(2)),Err(Error::Unauthorized));
        assert_eq!(check_owner(&s,Principal::anonymous()),Err(Error::Unauthorized));
        assert_eq!(check_owner(&s,s.owner),Ok(()));
    }

    #[test]
    fn model_replacement_rejects_stale_qtype_mapping() {
        let mut s=state();
        let mut manifest=laya_candle::pack::Manifest::parse(&s.upload.as_ref().unwrap().manifest).unwrap();
        let schema=schemas().remove(0); // Noul: exercise a changed primitive mapping.
        let compiled=schema::compile(schema.clone(),&FixtureTokenizer,manifest.primitive_to_qtype[1]).unwrap();
        let id=compiled.schema_hash;
        s.engine.register(compiled).unwrap();
        assert_eq!(check_schema_mapping(&s,&id),Ok(()));
        manifest.primitive_to_qtype.swap(1,2);
        let raw=serde_json::to_vec(&manifest).unwrap();
        s.engine.active_model=hash(&raw);
        s.upload.as_mut().unwrap().manifest=raw;
        assert_eq!(check_schema_mapping(&s,&id),Err(Error::BindingMismatch));
        // A new schema version can be registered using the new model's mapping.
        let mut revised=schema;
        revised.version+=1;
        let compiled=schema::compile(revised,&FixtureTokenizer,manifest.primitive_to_qtype[1]).unwrap();
        let revised_id=compiled.schema_hash;
        s.engine.register(compiled).unwrap();
        assert_eq!(check_schema_mapping(&s,&revised_id),Ok(()));
        s.engine.active_model=[0;32];
        assert_eq!(check_schema_mapping(&s,&revised_id),Err(Error::BindingMismatch));
        s.fixture_mode=true;
        assert_eq!(check_schema_mapping(&s,&id),Ok(()));
    }
}
