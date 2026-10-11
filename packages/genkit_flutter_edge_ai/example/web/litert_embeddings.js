import { m as A, d as E, s as S, r as $ } from "./tensorflow.js";
import { l as k, a as L, T as M } from "./litert.js";
import { S as I } from "./sentencepiece.js";
const z = 2, R = 2, y = 1, F = 0;
let r = null, s = null, u = !1, w = !1, c = null, f = null, p = null, a = null, d = null, m = null;
async function W(e) {
  try {
    const t = await fetch(e);
    if (!t.ok)
      throw new Error(`Failed to fetch tokenizer: ${t.status} ${t.statusText}`);
    const o = await t.arrayBuffer(), l = new I();
    if (typeof l.loadFromBuffer == "function")
      await l.loadFromBuffer(new Uint8Array(o));
    else {
      const n = new Uint8Array(o);
      let i = "";
      for (let T = 0; T < n.length; T++)
        i += String.fromCharCode(n[T]);
      const g = btoa(i);
      await l.loadFromB64StringModel(g);
    }
    return s = {
      encode: (n, i = !1) => {
        const g = l.encodeIds(n);
        return i ? [R, ...g, y] : g;
      },
      decode: (n) => l.decodeIds(n),
      processor: l
    }, s;
  } catch (t) {
    throw new Error("Failed to load SentencePiece tokenizer: " + t.message);
  }
}
async function _(e, t = "/node_modules/@litertjs/core/wasm/") {
  try {
    console.log(`[LiteRT] Loading model from: ${e}`), console.log(`[LiteRT] WASM loaded flag: ${w}`), await S("webgl"), await $(), w ? console.log("[LiteRT] WASM runtime already loaded, reusing") : (console.log(`[LiteRT] Loading WASM runtime from: ${t}`), await k(t), w = !0, console.log("[LiteRT] WASM runtime loaded successfully"));
    try {
      console.log("[LiteRT] Attempting to compile model with WebGPU..."), a = "webgpu", r = await L(e, {
        accelerator: "webgpu"
      }), console.log("[LiteRT] Model compiled, accelerator confirmed on first run");
    } catch (o) {
      console.warn("[LiteRT] WebGPU not available, falling back to WASM:", o.message), a = "wasm", r = await L(e, {
        accelerator: "wasm"
      }), console.log("[LiteRT] Model compiled with WASM successfully");
    }
    return d = r.options?.accelerator ?? null, d && d !== a && console.warn(
      `[LiteRT] Compiled for ${d}, not the requested ${a}. LiteRT recompiled without raising.`
    ), D(r), c = v(r), f = B(r), console.log(
      `[LiteRT] Model input length ${c}, dimension ${f}`
    ), r;
  } catch (o) {
    throw new Error("Failed to load LiteRT model: " + o.message);
  }
}
function v(e) {
  const t = e.getInputDetails()?.[0]?.shape, o = t && t.length >= 2 ? t[1] : void 0;
  if (!Number.isInteger(o) || o <= 0)
    throw new Error(
      `cannot read the input sequence length from the model (input 0 shape: ${h(t)})`
    );
  return o;
}
function B(e) {
  const t = e.getOutputDetails()?.[0]?.shape, o = t && t.length >= 1 ? t[t.length - 1] : void 0;
  if (!Number.isInteger(o) || o <= 0)
    throw new Error(
      `cannot read the embedding dimension from the model (output 0 shape: ${h(t)})`
    );
  return o;
}
function h(e) {
  return e ? `[${Array.from(e).join(", ")}]` : "none";
}
function b(e) {
  if (!u || !s)
    throw new Error("LiteRT embeddings not initialized. Call loadLiteRtEmbeddings first.");
  if (typeof e != "string")
    throw new Error(`The text must be a string, got ${typeof e}`);
  return [R, ...s.processor.encodeIds(e), y];
}
async function C(e) {
  const t = new Int32Array(c).fill(F);
  t.set(e);
  const o = new M(t, [1, c]);
  try {
    const n = (await r.run(o))[0];
    N(n.accelerator);
    let i = n;
    try {
      return n.accelerator === "webgpu" && (i = await n.moveTo("wasm")), Float32Array.from(i.toTypedArray());
    } finally {
      i !== n && !i.deleted && i.delete(), n.deleted || n.delete();
    }
  } finally {
    o.deleted || o.delete();
  }
}
function D(e) {
  if (d && d !== a) {
    m = !1;
    return;
  }
  let t;
  try {
    t = e.isFullyAccelerated;
  } catch {
    return;
  }
  const o = d ?? a;
  o !== "wasm" && (t === !1 ? (m = !1, console.warn(
    `[LiteRT] Model is not fully accelerated on ${o}. Unsupported ops run in WASM, so the accelerator reported after the first embedding is where the output buffer lives, not where every op ran.`
  )) : t === !0 && (m = !0));
}
function N(e) {
  p !== null || !e || (p = e, a && e !== a ? console.warn(
    `[LiteRT] Running on ${e}, not the requested ${a}. LiteRT fell back without raising — the model was not fully accelerated.`
  ) : console.log(`[LiteRT] Running on ${e}`));
}
window.liteRtEmbeddingsApiVersion = z;
window.loadLiteRtEmbeddings = async function(e, t, o) {
  try {
    if (u) {
      console.log("[LiteRT] Cleaning up previous model before reinitialization (hot restart detected)");
      try {
        await window.cleanupLiteRtEmbeddings();
      } catch (n) {
        console.warn("[LiteRT] Non-fatal cleanup error (will reinitialize anyway):", n), r = null, s = null, w = !1, u = !1, c = null, f = null;
      }
    }
    const l = o ?? "/node_modules/@litertjs/core/wasm/";
    return await W(t), await _(e, l), u = !0, { dimension: f, maxInputTokens: c };
  } catch (l) {
    throw u = !1, new Error("Failed to initialize LiteRT embeddings: " + l.message);
  }
};
window.embedLiteRtText = async function(e) {
  const t = b(e);
  return t.length > c ? {
    status: "too_long",
    actualTokens: t.length,
    maxInputTokens: c
  } : { status: "ok", values: await C(t) };
};
window.tokenizeLiteRtText = function(e) {
  return Int32Array.from(b(e));
};
window.getLiteRtEmbeddingAccelerator = function() {
  return p;
};
window.getLiteRtEmbeddingFullyAccelerated = function() {
  return m;
};
window.getLiteRtEmbeddingDimension = function() {
  return f;
};
window.cleanupLiteRtEmbeddings = async function() {
  if (p = null, a = null, d = null, m = null, console.log("[LiteRT] ========================================"), console.log("[LiteRT] Starting cleanup..."), console.log("[LiteRT] ========================================"), r)
    try {
      typeof r.delete == "function" && !r.deleted && (r.delete(), console.log("[LiteRT] ✅ Model deleted"));
    } catch (e) {
      console.warn("[LiteRT] ⚠️  Error deleting model (non-fatal):", e);
    }
  if (r = null, s)
    try {
      s.processor && typeof s.processor.delete == "function" && (s.processor.delete(), console.log("[LiteRT] ✅ Tokenizer deleted"));
    } catch (e) {
      console.warn("[LiteRT] ⚠️  Error deleting tokenizer (non-fatal):", e);
    }
  s = null;
  try {
    const t = A().numTensors;
    t > 0 && (console.log(`[LiteRT] Disposing ${t} TensorFlow.js tensors`), E(), console.log("[LiteRT] ✅ Tensors disposed"));
  } catch (e) {
    console.warn("[LiteRT] ⚠️  Error disposing tensors (non-fatal):", e);
  }
  console.log("[LiteRT] ✅ Keeping WASM runtime (reusable across models)"), c = null, f = null, u = !1, console.log("[LiteRT] ========================================"), console.log("[LiteRT] ✅ Cleanup completed"), console.log("[LiteRT] ========================================");
};
window.isLiteRtEmbeddingsInitialized = function() {
  return u;
};
console.log("LiteRT Embeddings module loaded successfully");
