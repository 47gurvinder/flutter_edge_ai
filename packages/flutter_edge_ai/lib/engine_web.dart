/// `dart:js_interop` helpers for the web arms of engine packages.
///
/// Import only from web arms (files selected by an
/// `if (dart.library.js_interop)` conditional import or export). The web-safe
/// engine SPI is `engine.dart`.
library;

export 'src/web/web_model_source.dart'
    show
        BlobUrlModelSource,
        OpfsStreamModelSource,
        WebModelSource,
        WebModelSourceResolver;
export 'src/web/web_model_url.dart' show resolveWebModelUrl;
