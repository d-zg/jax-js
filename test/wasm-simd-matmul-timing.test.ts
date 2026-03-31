/**
 * Timing test for WASM SIMD matmul — not for committing.
 *
 * Run with:
 *   npm run build && npx vitest run test/wasm-simd-matmul-timing.test.ts
 */
import { suite, test } from "vitest";
import { defaultDevice, init, jit, numpy as np, random, setDebug } from "@jax-js/jax";

const devices = await init();
setDebug(0);

suite.skipIf(!devices.includes("wasm"))("WASM SIMD timing", () => {
  defaultDevice("wasm");

  // Matmul
  for (const n of [32, 64, 128, 256, 512]) {
    test(`matmul ${n}x${n}`, async () => {
      const a = random.uniform(random.key(0), [n, n]);
      const b = random.uniform(random.key(1), [n, n]);

      const f = jit((a: np.Array, b: np.Array) => np.matmul(a, b));

      const warmup = f(a.ref, b.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 64 ? 1000 : 500;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(a.ref, b.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      const msPerCall = elapsed / N;
      const flops = 2 * n * n * n;
      const gflops = flops / (msPerCall / 1000) / 1e9;
      console.log(
        `[matmul ${n}x${n}] ${msPerCall.toFixed(3)}ms/call (${gflops.toFixed(2)} GFLOP/s, ${N} runs)`,
      );

      a.dispose();
      b.dispose();
    });
  }

  // Row sum (reduction, contiguous ridx → wide load)
  for (const n of [64, 128, 256, 512, 1024]) {
    test(`row sum ${n}x${n}`, async () => {
      const x = random.uniform(random.key(0), [n, n]);

      const f = jit((x: np.Array) => x.sum(1));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 128 ? 1000 : 500;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[row sum ${n}x${n}] ${(elapsed / N).toFixed(3)}ms/call (${N} runs)`,
      );

      x.dispose();
    });
  }

  // Row sum noncontiguous (reduction, contiguous ridx → wide load)
  for (const n of [64, 128, 256, 512, 1024]) {
    test(`row sum ${n}x${n}`, async () => {
      const y = random.uniform(random.key(0), [n, n]);
      const x = np.transpose(y.ref);

      const f = jit((x: np.Array) => x.sum(1));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 128 ? 1000 : 500;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[row sum noncontiguous ${n}x${n}] ${(elapsed / N).toFixed(3)}ms/call (${N} runs)`,
      );

      x.dispose();
    });
  }

  // Pointwise add
  for (const size of [1_000, 10_000, 100_000, 1_000_000]) {
    test(`pointwise add [${size.toLocaleString()}]`, async () => {
      const x = random.uniform(random.key(0), [size]);

      const f = jit((x: np.Array) => x.add(2).mul(3));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = size <= 10_000 ? 1000 : 500;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[pointwise add+mul ${size.toLocaleString()}] ${(elapsed / N).toFixed(3)}ms/call (${N} runs)`,
      );

      x.dispose();
    });
  }
});
