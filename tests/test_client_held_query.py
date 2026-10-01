import struct
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from client_held_query import QuerySession, decode_progress, encode_input, encode_continue
from measure_inference import Failure, blob
from profile_quantized_queries import unwrap_progress
from query_transport import AUTO_BUNDLE, MAX_STATE_BYTES, auto_width, unpack_state, inspect_state


def progress(state=None, step=0, done=False):
    body = 'logits = vec { 1.0 : float32; -2.0 : float32 }' if done else f'state = {blob(state)}'
    kind = "Done" if done else "Continue"
    return f'(variant {{ Ok = variant {{ {kind} = record {{ {body}; completed = {step} : nat32; total = 2 : nat32; instructions = 100 : nat64 }} }} }})'


def state(step):
    return b"LAYQ" + struct.pack("<II", 1, 1) + bytes(range(32)) + struct.pack("<IIII", step, 0, 4, 2) + b"\x00" * 128


class FakeIcp:
    env = "local"
    did = {"decision-engine": "engine.did"}

    def __init__(self):
        self.calls = 0
        self.fail = False
        self.payloads = []

    def run(self, args):
        self.payloads.append(Path(args[args.index("--args-file") + 1]).read_bytes())
        if self.fail:
            self.fail = False
            raise Failure("transport failed")
        self.calls += 1
        if self.calls <= 2:
            return progress(state(self.calls - 1), self.calls - 1)
        return progress(step=2, done=True)


class ClientHeldQueryTests(unittest.TestCase):
    def test_http_failures_retry_identical_bytes_and_remain_bounded(self):
        for status in (429, 502, 503, 504):
            class HttpOnce(FakeIcp):
                def __init__(self):
                    super().__init__(); self.failed = False
                def run(self, args):
                    if self.calls == 1 and not self.failed:
                        self.failed = True
                        self.payloads.append(Path(args[args.index("--args-file") + 1]).read_bytes())
                        raise Failure(f"HTTP Error: status: {status} Service Unavailable")
                    return super().run(args)
            icp = HttpOnce()
            with patch("client_held_query.time.sleep"):
                result = QuerySession(icp, dict(input_ids=[1,3,3,2], markers=[1,2], qtype_id=0)).finish()
            self.assertEqual(icp.payloads[1], icp.payloads[2])
            self.assertEqual(len(result["failed_query_attempts"]), 1)
        class AlwaysHttp(FakeIcp):
            def run(self, args):
                self.payloads.append(Path(args[args.index("--args-file") + 1]).read_bytes())
                raise Failure("HTTP 503 Service Unavailable")
        icp = AlwaysHttp()
        with patch("client_held_query.time.sleep"), self.assertRaises(Failure):
            QuerySession(icp, dict(input_ids=[1,3,3,2], markers=[1,2], qtype_id=0), retries=2).finish()
        self.assertEqual(len(icp.payloads), 3)
        self.assertTrue(all(p == icp.payloads[0] for p in icp.payloads))

    def test_permanent_http_error_does_not_retry_even_if_message_mentions_transport(self):
        class BadRequest(FakeIcp):
            def run(self, args):
                self.calls += 1
                raise Failure("transport HTTP Error: status code: 400 Bad Request")
        icp = BadRequest()
        with self.assertRaises(Failure):
            QuerySession(icp, dict(input_ids=[1,3,3,2], markers=[1,2], qtype_id=0)).finish()
        self.assertEqual(icp.calls, 1)

    def test_metadata_validates_entire_large_stream_without_unshuffling(self):
        raw = state(0)[:60] + b"\0" * (4 * 6 + 4 * 20000)
        offset = 60 + 4 * 6
        shuffled = raw[:offset] + b"".join(raw[offset+i::4] for i in range(4))
        wire = b"LAYZ" + struct.pack("<III", 1, 2, len(raw)) + zlib.compress(shuffled)
        self.assertEqual(inspect_state(wire), (raw[:60], len(raw)))
        with patch("query_transport.unpack_state", side_effect=AssertionError("full unpack called")):
            self.assertEqual(decode_progress(progress(wire))["state"], wire)
        corrupt = wire[:-1] + bytes([wire[-1] ^ 1])
        for broken in (corrupt, wire[:-1], wire+b"x",
                       wire[:12]+struct.pack("<I",len(raw)-1)+wire[16:],
                       wire[:12]+struct.pack("<I",MAX_STATE_BYTES+1)+wire[16:]):
            with self.assertRaises(Failure): inspect_state(broken)

    def test_fused_begin_completes_128_tokens_in_ten_queries(self):
        class Batched(FakeIcp):
            def __init__(self):
                super().__init__(); self.completed = 0
            def run(self, args):
                payload = Path(args[args.index("--args-file")+1]).read_bytes()
                self.payloads.append(payload)
                self.calls += 1
                if self.calls == 1:
                    self_test.assertEqual(args[3],"begin_token_inference_int8_batch_compressed_query")
                else:
                    self_test.assertEqual(args[3],"continue_token_inference_int8_compressed_query")
                self_test.assertIn("--query",args)
                width = struct.unpack("<I",payload[-4:])[0]
                self.completed = min(32,self.completed+width)
                raw = b"LAYI"+struct.pack("<II",1,2)+bytes.fromhex(AUTO_BUNDLE)+state(self.completed)[44:]
                return progress(raw,self.completed,self.completed==32).replace("total = 2 : nat32","total = 32 : nat32")
        self_test = self
        result = QuerySession(Batched(),dict(input_ids=[1]*128,markers=[1,2],qtype_id=0),steps=None,activation_format="int8",compression="auto").finish()
        self.assertEqual(result["inference_query_calls"],10)
        self.assertEqual(result["query_calls"][0]["completed"],3)
        self.assertTrue(result["fused_begin_query"])

    def test_fused_begin_limit_retries_same_input_with_smaller_width(self):
        class Limited(FakeIcp):
            def run(self,args):
                payload = Path(args[args.index("--args-file")+1]).read_bytes()
                self.payloads.append(payload)
                width = struct.unpack("<I",payload[-4:])[0]
                if width > 1: raise Failure("Canister exceeded instruction limit")
                raw = b"LAYI"+struct.pack("<II",1,2)+bytes.fromhex(AUTO_BUNDLE)+state(1)[44:]
                return progress(raw,1).replace("total = 2 : nat32","total = 32 : nat32")
        icp = Limited()
        session = QuerySession(icp,dict(input_ids=[1]*128,markers=[1,2],qtype_id=0),steps=None,activation_format="int8",compression="auto")
        self.assertEqual(session.advance()["completed"],1)
        self.assertEqual([struct.unpack("<I",p[-4:])[0] for p in icp.payloads],[3,1])
        self.assertEqual(icp.payloads[0][:-4],icp.payloads[1][:-4])
        self.assertEqual(session.auto_cap,1)
    def test_bounded_lossless_transport_and_opaque_retry_bytes(self):
        raw = b"LAYI" + struct.pack("<II", 1, 2) + state(0)[12:]
        wire = b"LAYZ" + struct.pack("<III", 1, 1, len(raw)) + zlib.compress(raw)
        decoded = decode_progress(progress(wire))
        self.assertEqual(decoded["state"], wire)
        self.assertEqual(unpack_state(wire), raw)
        self.assertEqual(decoded["activation_format"], "int8")
        for malformed in [wire[:-1], wire+b"x", wire[:4]+struct.pack("<I",2)+wire[8:],
                          wire[:12]+struct.pack("<I",MAX_STATE_BYTES+1)+wire[16:],
                          b"LAYZ"+struct.pack("<III",1,1,60)+zlib.compress(b"x"*(MAX_STATE_BYTES+1))]:
            with self.assertRaises(Failure): unpack_state(malformed)

    def test_f32_shuffle_restores_bits(self):
        raw = state(0)
        offset = 60 + 4*(4+2)
        raw = raw[:offset] + bytes(range(len(raw)-offset))
        shuffled = raw[:offset] + b"".join(raw[offset+i::4] for i in range(4))
        wire = b"LAYZ"+struct.pack("<III",1,2,len(raw))+zlib.compress(shuffled)
        self.assertEqual(unpack_state(wire), raw)

    def test_auto_widths_include_light_tail_without_excessive_encoder_batch(self):
        for tokens, mode, expected in [(128,"f32",3),(128,"int8",3),(65,"int8",6),(43,"int8",8),
                                       (66,"f32",6),(72,"f32",6),(73,"f32",4),(96,"f32",4),
                                       (97,"f32",3),(66,"int8",5),(72,"int8",5),(73,"int8",4),
                                       (90,"int8",4),(91,"int8",3)]:
            self.assertEqual(auto_width(tokens,mode,0),expected)
        self.assertEqual(auto_width(128,"int8",27),5)
        self.assertEqual(auto_width(128,"int8",26,2),2)
        self.assertEqual(auto_width(65,"int8",24),8)

    def test_tail_honors_reduced_cap_after_instruction_limit(self):
        class TailLimit(FakeIcp):
            def __init__(self):
                super().__init__()
                self.completed = 27
                self.widths = []
            def run(self, args):
                payload = Path(args[args.index("--args-file") + 1]).read_bytes()
                width = struct.unpack("<I", payload[-4:])[0]
                self.widths.append(width)
                if width > 1:
                    raise Failure("Canister exceeded instruction limit")
                self.completed += width
                raw = b"LAYI" + struct.pack("<II", 1, 2) + bytes.fromhex(AUTO_BUNDLE) + state(self.completed)[44:]
                return progress(raw, self.completed, self.completed == 32).replace("total = 2 : nat32", "total = 32 : nat32")
        icp = TailLimit()
        session = QuerySession(icp, dict(input_ids=[1]*128, markers=[1,2], qtype_id=0), steps=None, activation_format="int8")
        session.progress = dict(done=False, state=b"previous-state", completed=27, total=32)
        session.bundle = AUTO_BUNDLE
        result = session.finish()
        self.assertEqual(icp.widths, [5, 2, 1, 1, 1, 1, 1])
        self.assertEqual(len(result["failed_query_attempts"]), 2)
        self.assertEqual(session.auto_cap, 1)
        self.assertEqual(result["logits"], [1., -2.])
        for mode in ("f32", "int8"):
            for completed in range(27, 32):
                for cap in (1, 2, 3):
                    self.assertLessEqual(auto_width(128, mode, completed, cap), cap)

    def test_automatic_short_f32_uses_only_existing_query(self):
        class Short(FakeIcp):
            def run(self, args):
                self_test.assertEqual(args[3], "infer_tokens_query")
                self_test.assertIn("--query",args)
                self.calls += 1
                return '(variant { Ok = record { logits = vec { 1.0 : float32; -2.0 : float32 }; instructions = 100 : nat64 } })'
        self_test = self
        icp = Short()
        result = QuerySession(icp, dict(input_ids=[1,3,3,2],markers=[1,2],qtype_id=0),steps=None,compression="auto").finish()
        self.assertEqual(icp.calls,1)
        self.assertTrue(result["direct_short_query"])
        self.assertEqual(result["bundle_sha256"], AUTO_BUNDLE)
        self.assertEqual(result["response_continuation_bytes"],0)
    def test_profile_costs_do_not_override_handler_instruction_count(self):
        header = b"LAYI" + struct.pack("<II",1,2) + state(0)[12:] + b'{"}'
        reply = '(variant { Ok = record { costs = vec { record { instructions = 999 : nat64 } }; progress = ' + progress(header)[len('(variant { Ok = '):-len(' })')] + ' } })'
        decoded = decode_progress(unwrap_progress(reply))
        self.assertEqual(decoded["instructions"],100)
        self.assertEqual(decoded["state"],header)

    def test_int8_uses_separate_queries_and_retains_binary_state(self):
        class Int8Icp(FakeIcp):
            def run(self, args):
                method = args[3]
                self_test.assertIn(method, ("begin_token_inference_int8_query", "continue_token_inference_int8_query"))
                self_test.assertIn("--query", args)
                reply = super().run(args)
                q8 = b"LAYI" + struct.pack("<II",1,2) + state(self.calls - 1)[12:]
                return reply.replace(blob(state(self.calls - 1)), blob(q8)) if self.calls <= 2 else reply
        self_test = self
        icp = Int8Icp()
        result = QuerySession(icp, dict(input_ids=[1,3,3,2],markers=[1,2],qtype_id=0), activation_format="int8").finish()
        self.assertEqual(result["activation_format"], "int8")
        self.assertEqual(result["logits"], [1.,-2.])
        self.assertIn(b"LAYI",icp.payloads[1])

    def test_wrong_activation_format_does_not_replace_state(self):
        session = QuerySession(FakeIcp(), dict(input_ids=[1,3,3,2],markers=[1,2],qtype_id=0), activation_format="int8")
        with self.assertRaises(Failure): session.advance()
        self.assertIsNone(session.progress)

    def test_failed_call_preserves_state_for_exact_retry(self):
        icp = FakeIcp()
        session = QuerySession(icp, dict(input_ids=[1, 3, 3, 2], markers=[1, 2], qtype_id=0), retries=0)
        session.advance()
        before = session.progress
        icp.fail = True
        with self.assertRaises(Failure): session.advance()
        self.assertIs(session.progress, before)
        result = session.finish()
        self.assertEqual(icp.payloads[1], icp.payloads[2])
        self.assertEqual(result["logits"], [1.0, -2.0])
        self.assertEqual(result["inference_query_calls"], 3)
        self.assertEqual(result["instructions"], 300)
        with self.assertRaises(Failure): session.advance()

    def test_transport_errors_retry_identical_payload_automatically(self):
        icp = FakeIcp()
        session = QuerySession(icp, dict(input_ids=[1, 3, 3, 2], markers=[1, 2], qtype_id=0))
        session.advance()
        icp.fail = True
        result = session.finish()
        self.assertEqual(icp.payloads[1], icp.payloads[2])
        self.assertEqual(len(result["failed_query_attempts"]), 1)

    def test_two_phase_limit_retries_from_same_state_with_one_phase(self):
        class Limited(FakeIcp):
            def run(self, args):
                payload = Path(args[args.index("--args-file") + 1]).read_bytes()
                if self.calls == 1 and payload[-4:] == struct.pack("<I", 2):
                    self.payloads.append(payload)
                    raise Failure("Canister exceeded instruction limit")
                return super().run(args)
        icp = Limited()
        session = QuerySession(icp, dict(input_ids=[1, 3, 3, 2], markers=[1, 2], qtype_id=0), steps=2)
        result = session.finish()
        self.assertEqual(session.steps, 1)
        self.assertEqual(icp.payloads[1][:-4], icp.payloads[2][:-4])
        self.assertEqual(result["logits"], [1.0, -2.0])

    def test_large_batch_limit_halves_without_changing_continuation(self):
        class Limited(FakeIcp):
            def run(self, args):
                payload = Path(args[args.index("--args-file") + 1]).read_bytes()
                if self.calls == 1 and struct.unpack("<I", payload[-4:])[0] > 1:
                    self.payloads.append(payload)
                    raise Failure("Canister exceeded instruction limit")
                return super().run(args)
        icp = Limited()
        result = QuerySession(icp, dict(input_ids=[1, 3, 3, 2], markers=[1, 2], qtype_id=0), steps=8).finish()
        self.assertEqual([struct.unpack("<I", p[-4:])[0] for p in icp.payloads[1:5]], [8, 4, 2, 1])
        self.assertTrue(all(p[:-4] == icp.payloads[1][:-4] for p in icp.payloads[1:5]))
        self.assertEqual(len(result["failed_query_attempts"]), 3)
        self.assertEqual(result["logits"], [1., -2.])

    def test_progress_rejects_bad_revision_step_and_error(self):
        for offset in [0, 4, 8, 44]:
            broken = bytearray(state(0)); broken[offset:offset + 4] = b"\xff" * 4
            with self.assertRaises(Failure): decode_progress(progress(bytes(broken)))
        with self.assertRaises(Failure): decode_progress('(variant { Err = Unauthorized })')

    def test_binary_arguments_have_bounded_inputs_and_no_decimal_tensor_conversion(self):
        inp = dict(input_ids=[1, 3, 3, 2], markers=[1, 2], qtype_id=0)
        self.assertTrue(encode_input(inp).startswith(b"DIDL"))
        self.assertTrue(encode_continue(state(0), 1).endswith(state(0) + struct.pack("<I", 1)))
        for steps in [0, 17]:
            with self.assertRaises(Failure): encode_continue(state(0), steps)
        for changed in [dict(inp, input_ids=[]), dict(inp, input_ids=[1] * 129), dict(inp, qtype_id=3), dict(inp, markers=[-1, 2])]:
            with self.assertRaises(Failure): encode_input(changed)


if __name__ == "__main__":
    unittest.main()
