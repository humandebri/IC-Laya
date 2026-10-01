# IC-Laya: running typed AI decisions inside an Internet Computer canister

> Follow-up measurement: the historical BOOM DAO prompts have now also been run through the local canister's F32 and INT8 split-query paths. See [the new benchmark](BOOMDAO_QUERY_BENCHMARK.md). The measurements below preserve the earlier experiment.

I have been working on [IC-Laya](https://github.com/humandebri/IC-Laya), an independent Rust implementation that runs Laya inference inside an Internet Computer canister. The model computation executes as Wasm in the canister, with the model weights loaded there. It does not call an external inference API to obtain its answer.

The implementation has been tested with a real checkpoint on a **local IC network**. The measurements below describe that environment; mainnet deployment, operating cost, and latency still need to be evaluated.

Laya takes a question, context, and a set of possible answers, then scores those answers. IC-Laya supports three decision types: **Choice** for selecting among options, **Noul** for a boolean decision, and **Score** for evaluating ordered options. This gives a canister a way to classify short pieces of natural language and return a constrained result. For example, an application could ask which category a request belongs to, or whether a short report should be sent for human review. These are potential applications that would each need their own quality evaluation.

The reason I am exploring this on the IC is that an application can keep its model computation within the canister execution environment. For update calls, that computation can run under the IC's replicated execution model instead of depending on an external inference service. The model, input format, and application policy can be explicit parts of the system. This provides an execution property: it does not establish that the model's answer is correct. Input provenance, model quality, and the policy that consumes its output still matter.

The main engineering work has been fitting the model into the canister's resource budget. IC-Laya uses **W8A8 INT8 inference**: quantized weights and activations for the linear layers, with floating-point operations retained where needed. In the tested export, canonical weight storage fell from about **1.68 GB to 423 MB**. The runtime also provides a resumable inference path that carries intermediate state across update calls.

Here are the current local-canister measurements:

| Capability | Observed result | Scope |
| --- | --- | --- |
| Single update inference | A 128-token Choice input completed using approximately **39.248 billion instructions** | One measured input with a fixed Wasm and model pack |
| Resumable inference | A 128-token input completed across **two update calls** | Demonstrates the continuation path |
| Short raw queries | All 18 tested 16-token cases succeeded; the maximum was about **4.756 billion instructions** | The endpoint is restricted to the measured pack and at most 16 total tokens |

The 128-token input budget includes the question, answer options, separators, and context. The 16-token query path is therefore very limited for meaningful application questions. It returns raw logits, and its response is not certified. The current raw inference endpoints are owner-only. Measurement details and setup instructions are in the repository's [optimization notes](https://github.com/humandebri/IC-Laya/blob/main/docs/INT8_OPTIMIZATION_V4.md), [short-query notes](https://github.com/humandebri/IC-Laya/blob/main/docs/INT8_SHORT_QUERY.md), and [local setup guide](https://github.com/humandebri/IC-Laya/blob/main/docs/GETTING_STARTED.md).

There are also implementation checks against the upstream model. On four fixed inputs, the INT8 port selected the same top answer as the upstream implementation, with a maximum absolute logit difference of 0.141. That is a limited compatibility check, not a task-accuracy benchmark. A separate small local-canister probe matched the assigned labels on 14 of 16 handwritten English examples, which is also too small and unrepresentative to establish application quality. [Compatibility details](https://github.com/humandebri/IC-Laya/blob/main/docs/INT8.md) and [classification probe](https://github.com/humandebri/IC-Laya/blob/main/docs/INT8_PRACTICAL_128.md).

As a reference application, I also tried **SNS proposal review using historical BOOM DAO proposals**. I fetched public proposal data from the Dashboard API and tested short, structured summaries with the local native Laya backend. This was a separate application experiment; these BOOM DAO runs were not canister benchmarks.

Two examples illustrate the result:

| Historical proposal | Input fact | Laya result in the final supplementary prompt |
| --- | --- | --- |
| [617](https://dashboard.internetcomputer.org/sns/xjngq-yaaaa-aaaaq-aabha-cai/proposal/617) | The proposal-creation rendering shows the minimum voting dissolve delay changing from 1 day to 20,000 days, about 54.8 years | **“Unlikely”** to reduce voter participation or concentrate voting power |
| [653](https://dashboard.internetcomputer.org/sns/xjngq-yaaaa-aaaaq-aabha-cai/proposal/653) | Mint 250 million SNS tokens to one account | **“Likely”** to concentrate token control |

The miss on 617 is substantial. Other prompt experiments also changed their answers when the wording, units, or option order changed. I therefore cannot justify using the tested classifier as the sole authority for approving or blocking a proposal. These labels are uncalibrated outputs, not danger probabilities.

For comparison, I built a small local triage prototype that calculates numerical changes directly and keeps Laya's output as optional supplementary information. It flagged 617 for critical review even when Laya answered “unlikely.” That detection came from the numerical rule; it does not demonstrate added value from Laya. The rules and thresholds were developed after examining this case, and the four selected proposals do not establish a false-alert rate or recall.

The case also exposed a data problem: a proposal's creation-time snapshot can differ from the state when it executes. The [existing forum analysis](https://forum.dfinity.org/t/reviving-the-sns-framework-addressing-current-challenges-and-exploring-solutions/59196) describes an effective 48-hour minimum before 617 executed, and proposal 620 subsequently restoring that setting. A practical reviewer would need to follow the changing state across proposals. This experiment complements the discussion around [making DAO community settings critical](https://forum.dfinity.org/t/make-sns-topic-dao-community-settings-critical/46689); it does not establish that an alert would have prevented the historical outcome.

The project has demonstrated real-model inference inside a local IC canister and identified a usable execution path for short, typed decisions. Finding tasks where those decisions are reliable is the next step. The current executor demonstration uses a mock ledger, real-fund execution is disabled, and resumable raw inference is not yet connected to that executor workflow. Pretrained weights are supplied separately; the repository includes random-weight fixtures for implementation tests.

I would welcome feedback on short classification tasks that would benefit from canister-resident inference, representative datasets for evaluating them, and practical requirements for a mainnet pilot. SNS proposal review was one exploratory example. The broader question I want to explore is where a small, explicitly evaluated decision model can be useful inside an IC application.
