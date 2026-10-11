// SentencePiece's USER_DEFINED pieces, which `dart_sentencepiece_tokenizer`
// (1.4.1) does not honour.
//
// Google's SentencePiece matches a model's user-defined pieces BEFORE the model
// runs: at each position of the normalized text it takes the longest
// user-defined piece that starts there (its `PrefixMatcher`), and the BPE model
// treats that piece as one frozen symbol — never merged with a neighbour. The
// Dart package treats them as ordinary vocabulary, so Gemma's tag pieces come
// apart: `</div>` is the single piece 235 in SentencePiece (and in the LiteRT.js
// web bundle) but `</` `div` `>` (954, 1928, 236813) in the package, and
// `<mask>`, `<ctrl1>`, `[multimodal]` likewise. Gemma's vocabulary has 6,410 of
// them: 6,316 tags and 92 runs of spaces, newlines and tabs, which the
// package's BPE already happens to reach.
//
// [UserDefinedPieces.encode] does what SentencePiece does: the longest-match
// pass over the normalized text, each match emitted as its own id, and the
// text between two matches handed to the package. For BPE that is exact, not
// an approximation: a frozen symbol takes part in no merge, so every merge
// happens inside one stretch between two matches, in the same order as when
// the whole string is one agenda. And because the pass matches at EVERY
// position, greedily from the left, a stretch contains no user-defined piece
// for the package's BPE to build either.
//
// It applies to a pipeline where "normalize the whole text, then split" equals
// "split, then normalize each stretch": a BPE model whose normalization is
// character by character (at most spaces escaped to U+2581) — no dummy prefix,
// no whitespace collapsing, no precompiled character map. That is Gemma's, in
// all three file forms the loader reads. Anything else (a Unigram model such as
// Gecko, an NFKC map, a Metaspace pre-tokenizer) gets no pass and tokenizes
// exactly as before: the reads below return null for it.
//
// test/sentencepiece_reference_test.dart compares the result with the Python
// `sentencepiece` package row by row.

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';

import 'sentencepiece_model_proto.dart';

/// U+2581, what SentencePiece escapes a space to.
const _escapedSpace = '\u2581';

/// A model's user-defined pieces and how its normalizer treats spaces: what
/// [encode] needs to tokenize as SentencePiece does.
final class UserDefinedPieces {
  UserDefinedPieces._(Map<String, int> pieces, {required this.escapesSpaces})
    : _ids = pieces,
      _lengthsByFirstUnit = _lengths(pieces.keys);

  /// Piece → id.
  final Map<String, int> _ids;

  /// For each first UTF-16 code unit of a piece, the lengths of the pieces
  /// that start with it, longest first: the only substrings worth looking up.
  final Map<int, List<int>> _lengthsByFirstUnit;

  /// Whether the model's normalizer escapes ' ' to U+2581. The pieces are
  /// matched on the NORMALIZED text, so Gemma's run piece `\u2581\u2581`
  /// matches two spaces of input.
  final bool escapesSpaces;

  /// How many user-defined pieces the model has.
  int get length => _ids.length;

  static Map<int, List<int>> _lengths(Iterable<String> pieces) {
    final byUnit = <int, Set<int>>{};
    for (final piece in pieces) {
      (byUnit[piece.codeUnitAt(0)] ??= <int>{}).add(piece.length);
    }
    return {
      for (final MapEntry(:key, :value) in byUnit.entries)
        key: value.toList()..sort((a, b) => b - a),
    };
  }

  /// The ids SentencePiece produces for [text] (no BOS, no EOS): the longest
  /// user-defined piece at each position, as its own id, and everything
  /// between two of them encoded by [bpe] — the package tokenizer loaded from
  /// the same file, which must apply no added-token matching of its own (the
  /// loader guarantees that).
  ///
  /// Each stretch is handed over already normalized. That is a no-op for the
  /// pipelines this class is built for (escaping is the whole normalization,
  /// and a stretch has no space left to escape), and it is what keeps the
  /// matching and the encoding on the same characters.
  List<int> encode(SentencePieceTokenizer bpe, String text) {
    final normalized = escapesSpaces
        ? text.replaceAll(' ', _escapedSpace)
        : text;
    final ids = <int>[];
    var stretch = 0;
    var i = 0;
    while (i < normalized.length) {
      final match = _longestAt(normalized, i);
      if (match == null) {
        i += _codePointLength(normalized, i);
        continue;
      }
      if (stretch < i) {
        ids.addAll(bpe.encode(normalized.substring(stretch, i)).ids);
      }
      ids.add(match.id);
      i += match.length;
      stretch = i;
    }
    if (stretch < normalized.length) {
      ids.addAll(bpe.encode(normalized.substring(stretch)).ids);
    }
    return ids;
  }

  ({int id, int length})? _longestAt(String text, int start) {
    final lengths = _lengthsByFirstUnit[text.codeUnitAt(start)];
    if (lengths == null) return null;
    final left = text.length - start;
    for (final length in lengths) {
      if (length > left) continue;
      final id = _ids[text.substring(start, start + length)];
      if (id != null) return (id: id, length: length);
    }
    return null;
  }

  static int _codePointLength(String text, int i) {
    final unit = text.codeUnitAt(i);
    if (unit >= 0xD800 && unit <= 0xDBFF && i + 1 < text.length) {
      final next = text.codeUnitAt(i + 1);
      if (next >= 0xDC00 && next <= 0xDFFF) return 2;
    }
    return 1;
  }

  /// For a SentencePiece `.model`, read by [SentencePieceModelProto.parse].
  /// Null when the model has no user-defined piece, or is not a BPE model
  /// with character-by-character normalization (see the file header).
  static UserDefinedPieces? fromModelProto(SentencePieceModelProto model) {
    if (model.userDefined.isEmpty) return null;
    if (model.modelType != bpeModelType) return null;
    if (model.hasCharsmap ||
        model.addDummyPrefix ||
        model.removeExtraWhitespaces) {
      return null;
    }
    return UserDefinedPieces._(
      model.userDefined,
      escapesSpaces: model.escapeWhitespaces,
    );
  }

  /// Reads a HuggingFace `tokenizer.json` in Gemma's shape — a BPE model, the
  /// single `Replace(' ' → '\u2581')` normalizer, no pre-tokenizer or a
  /// `Split` on ' ' (a no-op once no space is left), `<pad>`/`<eos>`/`<bos>`
  /// at ids 0/1/2, and added tokens with no `lstrip`/`rstrip`/`single_word`/
  /// `normalized` flags. Null for any other file.
  ///
  /// A `tokenizer.json` carries no piece types. Its `added_tokens` are what the
  /// HuggingFace converter wrote from the SentencePiece model: every
  /// user-defined piece, plus `<pad>`, `<eos>`, `<bos>` and `<unk>`, which
  /// SentencePiece never produces from text, plus tokens outside the model's
  /// vocabulary (Gemma 3's `<image_soft_token>`, id 262144 of a 262,144-entry
  /// table). The user-defined pieces are therefore the added tokens that are
  /// none of those four and sit in the vocabulary under their own id.
  ///
  /// Returns, with them, the map to build the package tokenizer from: [json]
  /// with `added_tokens` emptied, because the package would otherwise match
  /// all of them in the text between two pieces — turning a literal `<bos>`
  /// into id 2 and `<image_soft_token>` into an id past the embedding table.
  static ({UserDefinedPieces pieces, Map<String, dynamic> tokenizerJson})?
  fromHuggingFaceJson(Map<String, dynamic> json) {
    final model = json['model'];
    if (model is! Map<String, dynamic> || model['type'] != 'BPE') return null;
    final vocab = model['vocab'];
    if (vocab is! Map<String, dynamic>) return null;
    if (!_isSpaceEscape(json['normalizer'])) return null;
    if (!_isNoOpAfterEscape(json['pre_tokenizer'])) return null;
    if (vocab['<pad>'] != 0 || vocab['<eos>'] != 1 || vocab['<bos>'] != 2) {
      return null;
    }
    final unk = model['unk_token'];
    final notFromText = <int>{0, 1, 2, if (vocab[unk] case final int id) id};

    final added = json['added_tokens'];
    final pieces = <String, int>{};
    if (added is List) {
      for (final token in added) {
        if (token is! Map<String, dynamic>) return null;
        final id = token['id'];
        final content = token['content'];
        if (id is! int || content is! String) return null;
        if (token['lstrip'] == true ||
            token['rstrip'] == true ||
            token['single_word'] == true ||
            token['normalized'] == true) {
          return null;
        }
        if (notFromText.contains(id) || vocab[content] != id) continue;
        if (content.isEmpty) continue;
        pieces[content] = id;
      }
    } else if (added != null) {
      return null;
    }
    return (
      pieces: UserDefinedPieces._(pieces, escapesSpaces: true),
      tokenizerJson: {...json, 'added_tokens': const <Object?>[]},
    );
  }

  /// Reads the package's own JSON format (`SentencePieceTokenizerJson.toJson`),
  /// which keeps the piece types. Null unless it is a BPE model with
  /// character-by-character normalization and at least one user-defined piece.
  static UserDefinedPieces? fromLibraryJson(Map<String, dynamic> json) {
    if (json['model_type'] != 'bpe' || json['pre_tokenizer'] != null) {
      return null;
    }
    final normalizer = json['normalizer'];
    if (normalizer is! Map<String, dynamic> ||
        normalizer['add_dummy_prefix'] != false ||
        normalizer['remove_extra_whitespaces'] != false ||
        normalizer['precompiled_charsmap'] != null ||
        normalizer['operations'] != null) {
      return null;
    }
    final vocab = json['vocab'];
    if (vocab is! Map<String, dynamic>) return null;
    final pieces = vocab['pieces'];
    final types = vocab['types'];
    if (pieces is! List || types is! List || pieces.length != types.length) {
      return null;
    }
    final userDefined = <String, int>{};
    for (var id = 0; id < types.length; id++) {
      final piece = pieces[id];
      if (types[id] != userDefinedPieceType ||
          piece is! String ||
          piece.isEmpty) {
        continue;
      }
      userDefined.putIfAbsent(piece, () => id);
    }
    if (userDefined.isEmpty) return null;
    return UserDefinedPieces._(
      userDefined,
      escapesSpaces: normalizer['escape_whitespaces'] != false,
    );
  }

  /// `{"type": "Replace", "pattern": {"String": " "}, "content": "\u2581"}`,
  /// alone or as the only member of a `Sequence`.
  static bool _isSpaceEscape(Object? normalizer) => switch (normalizer) {
    {'type': 'Sequence', 'normalizers': [final only]} => _isSpaceEscape(only),
    {'type': 'Replace', 'pattern': {'String': ' '}, 'content': _escapedSpace} =>
      true,
    _ => false,
  };

  /// No pre-tokenizer, or a `Split` on a literal space that is not inverted:
  /// after the escape no space is left for it to split on.
  static bool _isNoOpAfterEscape(Object? preTokenizer) =>
      switch (preTokenizer) {
        null => true,
        {'type': 'Split', 'pattern': {'String': ' '}, 'invert': final invert} =>
          invert != true,
        {'type': 'Split', 'pattern': {'String': ' '}} => true,
        _ => false,
      };
}
