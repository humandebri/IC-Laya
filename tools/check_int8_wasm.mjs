import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const path = process.argv[2];
assert.ok(path, 'usage: node tools/check_int8_wasm.mjs <int8_wasm_check.wasm>');
const module = await WebAssembly.compile(readFileSync(path));
assert.deepEqual(WebAssembly.Module.imports(module), [], 'harness must be standalone');
const instance = await WebAssembly.instantiate(module);
assert.equal(instance.exports.check(), 920);
assert.equal(instance.exports.check_quantization(), 144);
assert.equal(instance.exports.check_writeback(), 216);
assert.equal(instance.exports.check_q8_ops(), 144);
assert.equal(instance.exports.check_q8_statistics(), 24);
console.log('PASS: 920 integer + 144 quantization + 216 F32 writeback + 144 INT8 preparation/norm + 24 statistics Wasm cases');
