// What both `.litertlm` embedding passes share — the native one over
// LiteRT-LM's C API and the web one over `@litert-lm/core`'s EmbeddingEngine:
// the engine tag, the input cap, the audio refusal and the image containers.
// Web-safe: imports only core's engine library, so the web arm of
// `LiteRtEmbeddingBackend` can use it without pulling dart:ffi.

import 'package:flutter_edge_ai/engine.dart';

/// The engine's name in messages and errors, native and web alike: the same
/// LiteRT-LM runtime, compiled to WebAssembly on the web.
const liteRtLmEngineTag = 'LiteRT-LM';

/// This version's answer to declared audio, from `describe()` and from the
/// passes.
const liteRtLmNoAudioRefusal =
    'flutter_edge_ai_litertlm 1.15.0 embeds text and images; audio arrives in '
    '1.16.0';

/// The explicit text input cap (D21), on every platform. Never raised on
/// mobile; on the web it is also the runtime's own default, which keeps the
/// engine inside the 4 GB wasm32 address space.
const liteRtLmMaxInputLength = 1024;

/// The image containers LiteRT-LM's stb decoder handles, as measured on our
/// native-v0.18.0-d dylib (X3): PNG, JPEG, GIF and BMP decode; WebP, TIFF
/// and garbage fail with code 3; stb has no HEIC decoder. Reported at load, so
/// core refuses every other container on the caller's isolate, before the
/// request is sent.
///
/// The web runtime is the same C++ compiled to WebAssembly; its MIME list also
/// names PSD, HDR, PGM/PPM and TGA, but those were never measured and core's
/// sniffer does not recognise them, so the web pass reports this same set.
final liteRtLmImageFormats = Set<EncodedMediaFormat>.unmodifiable({
  EncodedMediaFormat.jpeg,
  EncodedMediaFormat.png,
  EncodedMediaFormat.gif,
  EncodedMediaFormat.bmp,
});

/// What the web arm says to `npu` for a `.litertlm` bundle: the NPU runs
/// EmbeddingGemma 2 only through the native Qualcomm dispatch.
const liteRtLmWebNpuRefusal =
    'EmbeddingGemma 2 runs on the NPU on Qualcomm Android and Qualcomm Linux '
    'arm64 only, not on the web. Omit preferredBackend here.';
