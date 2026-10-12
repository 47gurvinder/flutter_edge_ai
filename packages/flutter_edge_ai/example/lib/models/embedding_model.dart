import 'base_model.dart';

/// Vision tokens per EmbeddingGemma 2 image in this app: the
/// `LiteRtEmbeddingBackend` default, registered in gemma_bootstrap.dart.
const kExampleVisionTokensPerImage = 140;

/// The profile id of an image index over [textProfileId]'s embedding space:
/// image vectors also depend on the vision budget (R5 §5.6).
String imageIndexProfileId(String textProfileId) =>
    '$textProfileId-vis$kExampleVisionTokensPerImage';

enum EmbeddingModel implements EmbeddingModelInterface {
  // EmbeddingGemma-300M models (all generate 768D embeddings)
  // Numbers in names indicate max sequence length, not embedding dimension
  embeddingGemma1024(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/embeddinggemma-300M_seq1024_mixed-precision.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/sentencepiece.model',
    filename:
        'embeddinggemma-300M_seq1024_mixed-precision__rev-29888fcee321.tflite',
    tokenizerFilename:
        'embeddinggemma-300m-seq1024__sentencepiece__rev-29888fcee321.model',
    displayName: 'EmbeddingGemma (seq=1024)',
    size: '183MB',
    dimension: 768, // Fixed embedding dimension for EmbeddingGemma-300M
    maxSeqLen: 1024,
    needsAuth: true,
    ragProfileId:
        'embeddinggemma-300m-seq1024-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1',
  ),

  embeddingGemma2048(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/embeddinggemma-300M_seq2048_mixed-precision.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/sentencepiece.model',
    filename:
        'embeddinggemma-300M_seq2048_mixed-precision__rev-29888fcee321.tflite',
    tokenizerFilename:
        'embeddinggemma-300m-seq2048__sentencepiece__rev-29888fcee321.model',
    displayName: 'EmbeddingGemma (seq=2048)',
    size: '196MB',
    dimension: 768, // Fixed embedding dimension for EmbeddingGemma-300M
    maxSeqLen: 2048,
    needsAuth: true,
    ragProfileId:
        'embeddinggemma-300m-seq2048-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1',
  ),

  embeddingGemma256(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/embeddinggemma-300M_seq256_mixed-precision.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/sentencepiece.model',
    filename:
        'embeddinggemma-300M_seq256_mixed-precision__rev-29888fcee321.tflite',
    tokenizerFilename:
        'embeddinggemma-300m-seq256__sentencepiece__rev-29888fcee321.model',
    displayName: 'EmbeddingGemma (seq=256)',
    size: '179MB',
    dimension: 768, // Fixed embedding dimension for EmbeddingGemma-300M
    maxSeqLen: 256,
    needsAuth: true,
    ragProfileId:
        'embeddinggemma-300m-seq256-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1',
  ),

  // Local model for fast testing (no auth required)
  // Files are in example/assets/models/
  // AssetSource works in both debug and production builds
  localEmbeddingGemma256(
    url: 'assets/models/embeddinggemma-300M_seq256_mixed-precision.tflite',
    tokenizerUrl: 'assets/models/sentencepiece.model',
    filename:
        'embeddinggemma-300M_seq256_mixed-precision__example-asset-v1.tflite',
    tokenizerFilename:
        'embeddinggemma-300m-seq256__sentencepiece__example-asset-v1.model',
    displayName: '🚀 Local EmbeddingGemma (seq=256)',
    size: '171MB (Local - No Auth)',
    dimension: 768, // Fixed embedding dimension for EmbeddingGemma-300M
    maxSeqLen: 256,
    needsAuth: false,
    sourceType: ModelSourceType
        .asset, // Use Flutter assets - works in debug and production
    // Kept separate from the hosted seq=256 profile until the checked-in asset
    // is proven byte-identical to that immutable Hugging Face revision.
    ragProfileId:
        'embeddinggemma-300m-seq256-mp-example-asset-v1-retrieval-prefix-meanpool-l2-v1',
  ),

  embeddingGemma512(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/embeddinggemma-300M_seq512_mixed-precision.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/embeddinggemma-300m/resolve/29888fcee3216acadc7e844906e5fe0d79a61875/sentencepiece.model',
    filename:
        'embeddinggemma-300M_seq512_mixed-precision__rev-29888fcee321.tflite',
    tokenizerFilename:
        'embeddinggemma-300m-seq512__sentencepiece__rev-29888fcee321.model',
    displayName: 'EmbeddingGemma (seq=512)',
    size: '179MB',
    dimension: 768, // Fixed embedding dimension for EmbeddingGemma-300M
    maxSeqLen: 512,
    needsAuth: true,
    ragProfileId:
        'embeddinggemma-300m-seq512-mp-rev-29888fcee321-retrieval-prefix-meanpool-l2-v1',
  ),

  // EmbeddingGemma 2 text (270M): ONE self-contained `.litertlm` bundle that
  // carries its own tokenizer, so there is no tokenizer file. Ungated.
  // Android, iOS, macOS, Linux, Windows (CPU), and the web (WebGPU, since
  // flutter_edge_ai_litertlm 1.15.0; no CPU path there). Input cap 1024 tokens
  // (prefix, BOS and EOS included) — keep RAG chunks at 512 or below.
  embeddingGemma2Text270m(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/9be6e8b90982095dc05c2bd162e4b954ee4dbac7/embeddinggemma-2-text-270m.litertlm',
    filename: 'embeddinggemma-2-text-270m__rev-9be6e8b90982.litertlm',
    displayName: 'EmbeddingGemma 2 Text (270M)',
    size: '165MB',
    dimension: 768,
    maxSeqLen: 1024,
    needsAuth: false,
    sha256: '2d079ee2f6f066b1f368e8d7c819f55214eaef1d0513b312321901f30ab286fb',
    // Keyed by the text-encoder weights section (sha256 prefix 4410a540ce75):
    // byte-identical in the 270m, 440m and 740m generic bundles, which give
    // identical text vectors, so one id serves all three. Never shared with
    // EmbeddingGemma 1 or an ONNX export of EG2.
    ragProfileId: 'embeddinggemma2-litertlm-text-4410a540ce75-prefix-v1',
  ),

  // EmbeddingGemma 2 Text + Vision (440M): the 270M's text tower (the same
  // bytes, so the same text vectors and the same text profile id) plus a
  // vision encoder, so it embeds images too — the Image Search screen needs
  // it. Ungated; every platform, the web included. What the images cost over
  // the 270M (macOS figures): 388 MB on disk instead of 165, a 218 MB vision
  // weight cache written at every load, even for text-only use (native; the
  // web keeps no weight cache), and about 385 MB more RAM from the first image
  // on. For text alone the 270M is the better choice.
  embeddingGemma2TextVision440m(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm/resolve/e301f74d5551b0c2641bd5cb4652a76239d5c5f8/embeddinggemma-2-text-vision-440m.litertlm',
    filename: 'embeddinggemma-2-text-vision-440m__rev-e301f74d5551.litertlm',
    displayName: 'EmbeddingGemma 2 Text + Vision (440M)',
    size: '388MB',
    dimension: 768,
    maxSeqLen: 1024,
    needsAuth: false,
    sha256: '92dcbea108899e5d6e30d919b0744f90d9967e80c67a4ab5503ac16d54f62eb0',
    // Text vectors are identical to the 270M's, so text stores share its id.
    // An image index uses imageRagProfileId.
    ragProfileId: 'embeddinggemma2-litertlm-text-4410a540ce75-prefix-v1',
    embedsImages: true,
  ),

  // EmbeddingGemma 2 (740M): the 440M's text and vision towers (the same
  // bytes, so the same text and image vectors and the same profile ids) plus
  // an audio encoder, so it also embeds audio (WAV, MP3, FLAC; about 25
  // tokens per second, so a clip of up to about 40 s fits the 1024-token cap)
  // and combinations — a photo with its caption, a clip with a note — in one
  // vector; both native-only (the web refuses supportAudio). Ungated. On top
  // of the 440M (macOS figures): 485 MB on disk, a further 95 MB audio weight
  // cache written at every load (385 MB of caches in all, even for text-only
  // use), and more RAM on the first clip. Audio vectors do not depend on the
  // vision budget, so an audio index keeps the text id.
  embeddingGemma2TextVisionAudio740m(
    url:
        'https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/24d962e906c7d332c6428e71c9676855024569e2/embeddinggemma-2-740m.litertlm',
    filename: 'embeddinggemma-2-740m__rev-24d962e906c7.litertlm',
    displayName: 'EmbeddingGemma 2 Text + Vision + Audio (740M)',
    size: '485MB',
    dimension: 768,
    maxSeqLen: 1024,
    needsAuth: false,
    sha256: 'e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c',
    ragProfileId: 'embeddinggemma2-litertlm-text-4410a540ce75-prefix-v1',
    embedsImages: true,
    embedsAudio: true,
  ),

  // Gecko-110m models (generate 768D embeddings)
  // Gecko 64 is the smallest and fastest model - ideal for short queries
  gecko64(
    url:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/Gecko_64_quant.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/sentencepiece.model',
    filename: 'Gecko_64_quant__rev-61a0d0c2cdc9.tflite',
    tokenizerFilename:
        'gecko-110m-en-seq64__sentencepiece__rev-61a0d0c2cdc9.model',
    displayName: 'Gecko (seq=64)',
    size: '110MB',
    dimension: 768, // Fixed embedding dimension for Gecko-110m
    maxSeqLen: 64,
    needsAuth: false,
    ragProfileId:
        'gecko-110m-en-seq64-quant-rev-61a0d0c2cdc9-retrieval-prefix-meanpool-l2-v1',
  ),

  gecko256(
    url:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/Gecko_256_quant.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/sentencepiece.model',
    filename: 'Gecko_256_quant__rev-61a0d0c2cdc9.tflite',
    tokenizerFilename:
        'gecko-110m-en-seq256__sentencepiece__rev-61a0d0c2cdc9.model',
    displayName: 'Gecko (seq=256)',
    size: '114MB',
    dimension: 768, // Fixed embedding dimension for Gecko-110m
    maxSeqLen: 256,
    needsAuth: false,
    ragProfileId:
        'gecko-110m-en-seq256-quant-rev-61a0d0c2cdc9-retrieval-prefix-meanpool-l2-v1',
  ),

  gecko512(
    url:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/Gecko_512_quant.tflite',
    tokenizerUrl:
        'https://huggingface.co/litert-community/Gecko-110m-en/resolve/61a0d0c2cdc9b4f2c1727e63acb7ad86e68508c2/sentencepiece.model',
    filename: 'Gecko_512_quant__rev-61a0d0c2cdc9.tflite',
    tokenizerFilename:
        'gecko-110m-en-seq512__sentencepiece__rev-61a0d0c2cdc9.model',
    displayName: 'Gecko (seq=512)',
    size: '116MB',
    dimension: 768, // Fixed embedding dimension for Gecko-110m
    maxSeqLen: 512,
    needsAuth: false,
    ragProfileId:
        'gecko-110m-en-seq512-quant-rev-61a0d0c2cdc9-retrieval-prefix-meanpool-l2-v1',
  );

  /// Enum fields
  @override
  final String url;
  @override
  final String? tokenizerUrl;
  @override
  final String filename;
  @override
  final String? tokenizerFilename;
  @override
  final String displayName;
  @override
  final String size;
  @override
  final int dimension;
  @override
  final int maxSeqLen;
  @override
  final bool needsAuth;
  @override
  final ModelSourceType sourceType;

  /// Stable identity of the complete embedding space used by the RAG demo.
  ///
  /// Bump this whenever weights, tokenizer, task prefixes, pooling, or
  /// normalization change. It must never be derived from a mutable URL, local
  /// path, or vector dimension.
  final String ragProfileId;

  /// SHA-256 of the pinned model file, where the catalog records one. The
  /// EmbeddingGemma 2 integration suite checks the installed bundle against
  /// it.
  final String? sha256;

  /// Whether the model embeds images (a bundle with a vision encoder).
  final bool embedsImages;

  /// Whether the model embeds audio (a bundle with an audio encoder; native
  /// platforms only).
  final bool embedsAudio;

  /// The profile of an IMAGE index built with this model, or null when it
  /// embeds no images: the text id plus the vision budget the example
  /// registers (`-vis140`). Image vectors depend on that budget, text vectors
  /// do not, so text stores keep [ragProfileId].
  String? get imageRagProfileId =>
      embedsImages ? imageIndexProfileId(ragProfileId) : null;

  /// Constructor
  const EmbeddingModel({
    required this.url,
    this.tokenizerUrl,
    required this.filename,
    this.tokenizerFilename,
    required this.displayName,
    required this.size,
    required this.dimension,
    required this.maxSeqLen,
    required this.needsAuth,
    required this.ragProfileId,
    this.sha256,
    this.embedsImages = false,
    this.embedsAudio = false,
    this.sourceType = ModelSourceType
        .network, // Default to network for backward compatibility
  }) : assert(
         (tokenizerUrl == null) == (tokenizerFilename == null),
         'a tokenizer has both a URL and a file name, or neither',
       );

  /// A self-contained `.litertlm` bundle: no tokenizer file, and no web
  /// engine yet.
  bool get isLiteRtLmBundle => tokenizerUrl == null;

  // BaseModel interface implementation
  @override
  String get name => toString().split('.').last;

  @override
  ModelKind get kind => ModelKind.embedding;

  @override
  String? get licenseUrl => null; // Most embedding models don't have specific license URLs
}
