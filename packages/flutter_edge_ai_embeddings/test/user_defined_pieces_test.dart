@TestOn('vm')
library;

// The user-defined pass (`src/user_defined_pieces.dart`) on toy models small
// enough to reason about, in all three file forms the loader reads: a
// SentencePiece `.model`, a HuggingFace `tokenizer.json` and the dependency's
// own JSON. The real vocabularies are sentencepiece_reference_test.dart's.
//
// The toy vocabulary has the user-defined pieces `<a>`, `<ab>`, `▁▁` and an
// emoji, and the normal piece `x<` with the best score of all — the merge the
// dependency makes in `x<a>` when nothing freezes `<a>`.

import 'dart:convert';
import 'dart:io';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:flutter_edge_ai_embeddings/src/embedding_tokenizer.dart';
import 'package:flutter_edge_ai_embeddings/src/sentencepiece_model_proto.dart';
import 'package:flutter_edge_ai_embeddings/src/user_defined_pieces.dart';
import 'package:flutter_test/flutter_test.dart';

import 'proto_builder.dart';

final _space = String.fromCharCode(0x2581);
final _emoji = String.fromCharCodes([0xD83D, 0xDE00]);

/// (text, type, score); ids are the indices.
final _toyPieces = [
  ('<pad>', 3, 0.0),
  ('<eos>', 3, 0.0),
  ('<bos>', 3, 0.0),
  ('<unk>', 2, 0.0),
  ('<a>', userDefinedPieceType, 0.0), // 4
  ('<ab>', userDefinedPieceType, 0.0), // 5
  ('$_space$_space', userDefinedPieceType, 0.0), // 6
  (_emoji, userDefinedPieceType, 0.0), // 7
  ('a', 1, -1.0), // 8
  ('b', 1, -2.0), // 9
  ('<', 1, -3.0), // 10
  ('>', 1, -4.0), // 11
  (_space, 1, -5.0), // 12
  ('x', 1, -6.0), // 13
  ('x<', 1, -0.5), // 14
  ('ab', 1, -0.7), // 15
];

List<int> _toyModel({
  int modelType = bpeModelType,
  bool dummyPrefix = false,
}) => [
  for (final (text, type, score) in _toyPieces)
    ...piece(text, score: score, type: type),
  ...trainerSpec(modelType: modelType, unkId: 3, bosId: 2, eosId: 1, padId: 0),
  ...normalizerSpec(
    name: 'identity',
    addDummyPrefix: dummyPrefix,
    removeExtraWhitespaces: false,
    escapeWhitespaces: true,
  ),
];

/// The toy vocabulary as a HuggingFace `tokenizer.json`, the way the converter
/// writes one: every user-defined piece and the four specials as added tokens,
/// plus `<img>`, an added token outside the vocabulary.
Map<String, dynamic> _toyHfJson() => {
  'version': '1.0',
  'truncation': null,
  'padding': null,
  'added_tokens': [
    for (final (id, content, special) in [
      (0, '<pad>', true),
      (1, '<eos>', true),
      (2, '<bos>', true),
      (3, '<unk>', true),
      (4, '<a>', false),
      (5, '<ab>', false),
      (6, '$_space$_space', false),
      (7, _emoji, false),
      (16, '<img>', true),
    ])
      {
        'id': id,
        'content': content,
        'single_word': false,
        'lstrip': false,
        'rstrip': false,
        'normalized': false,
        'special': special,
      },
  ],
  'normalizer': {
    'type': 'Replace',
    'pattern': {'String': ' '},
    'content': _space,
  },
  'pre_tokenizer': {
    'type': 'Split',
    'pattern': {'String': ' '},
    'behavior': 'MergedWithPrevious',
    'invert': false,
  },
  'post_processor': null,
  'decoder': null,
  'model': {
    'type': 'BPE',
    'unk_token': '<unk>',
    'vocab': {for (final (id, (text, _, _)) in _toyPieces.indexed) text: id},
    'merges': [
      ['x', '<'],
      ['a', 'b'],
      [_space, _space],
    ],
  },
};

late Directory _dir;

Future<String> _write(String name, List<int> bytes) async {
  final file = File('${_dir.path}/$name');
  await file.writeAsBytes(bytes);
  return file.path;
}

/// Content ids (BOS and EOS stripped) of [text] through [encodeForEmbedding].
List<int> _content(SentencePieceTokenizer tokenizer, String text) {
  final ids = encodeForEmbedding(tokenizer, '', text);
  expect(ids.first, 2);
  expect(ids.last, 1);
  return ids.sublist(1, ids.length - 1);
}

void main() {
  setUp(() async {
    _dir = await Directory.systemTemp.createTemp('user_defined_pieces_test');
  });
  tearDown(() async {
    await _dir.delete(recursive: true);
  });

  /// What every form must produce (SentencePiece's answer, by hand).
  void expectSentencePieceIds(SentencePieceTokenizer tokenizer) {
    // `<a>` is frozen: `x` and `<` never merge into the best-scored `x<`.
    expect(_content(tokenizer, 'x<a>'), [13, 4]);
    // Longest match: `<ab>`, not `<a>` + `b>`.
    expect(_content(tokenizer, '<ab>'), [5]);
    expect(_content(tokenizer, '<a>b'), [4, 9]);
    expect(_content(tokenizer, '<a><a>'), [4, 4]);
    // Spaces are escaped before matching: two spaces are the `▁▁` piece.
    expect(_content(tokenizer, 'a  <a>'), [8, 6, 4]);
    // A surrogate pair is one character, and can be a piece.
    expect(_content(tokenizer, 'x$_emoji$_emoji'), [13, 7, 7]);
    // A piece that is not complete is ordinary text.
    expect(_content(tokenizer, '<a'), [10, 8]);
    expect(_content(tokenizer, ''), isEmpty);
  }

  group('a SentencePiece .model', () {
    test('BPE with only spaces escaped: user-defined pieces are matched as '
        'SentencePiece does', () async {
      final tokenizer = await loadEmbeddingTokenizer(
        await _write('toy.model', _toyModel()),
      );
      expectSentencePieceIds(tokenizer);
      // Without the pass the dependency merges into the frozen piece.
      expect(tokenizer.encode('x<a>').ids, [14, 8, 11]);
    });

    test(
      'a Unigram model gets no pass: the dependency\'s ids, unchanged',
      () async {
        final path = await _write('unigram.model', _toyModel(modelType: 1));
        final tokenizer = await loadEmbeddingTokenizer(path);
        expect(
          UserDefinedPieces.fromModelProto(
            SentencePieceModelProto.parse(File(path).readAsBytesSync()),
          ),
          isNull,
        );
        for (final text in ['x<a>', '<ab>', 'a  b']) {
          expect(_content(tokenizer, text), tokenizer.encode(text).ids);
        }
      },
    );

    test('a model with a dummy prefix gets no pass either', () async {
      final path = await _write('dummy.model', _toyModel(dummyPrefix: true));
      expect(
        UserDefinedPieces.fromModelProto(
          SentencePieceModelProto.parse(File(path).readAsBytesSync()),
        ),
        isNull,
      );
    });
  });

  group('a HuggingFace tokenizer.json in Gemma\'s shape', () {
    test('matches the same pieces, and turns no literal special token or '
        'out-of-vocabulary added token into its id', () async {
      final tokenizer = await loadEmbeddingTokenizer(
        await _write('tokenizer.json', utf8.encode(jsonEncode(_toyHfJson()))),
      );
      expectSentencePieceIds(tokenizer);
      for (final text in ['<bos>', '<eos>', '<pad>', '<unk>', 'x<img>']) {
        final content = _content(tokenizer, text);
        expect(content, isNot(contains(anyOf(0, 1, 2, 16))), reason: text);
      }
    });

    test('the returned tokenizer keeps Gemma\'s special-token ids', () async {
      final tokenizer = await loadEmbeddingTokenizer(
        await _write('tokenizer.json', utf8.encode(jsonEncode(_toyHfJson()))),
      );
      expect(tokenizer.encode('a', addSpecialTokens: true).ids, [2, 8, 1]);
    });

    test('the pieces are the added tokens minus the four specials and what '
        'is not in the vocabulary; the map to load has no added tokens, and '
        'the input map is untouched', () {
      final json = _toyHfJson();
      final read = UserDefinedPieces.fromHuggingFaceJson(json)!;
      expect(read.pieces.length, 4);
      expect(read.pieces.escapesSpaces, isTrue);
      expect(read.tokenizerJson['added_tokens'], isEmpty);
      expect(read.tokenizerJson['model'], same(json['model']));
      expect(json['added_tokens'], hasLength(9));
    });

    for (final (name, change)
        in <(String, void Function(Map<String, dynamic>))>[
          ('a Unigram model', (j) => (j['model'] as Map)['type'] = 'Unigram'),
          ('no normalizer', (j) => j['normalizer'] = null),
          (
            'a Metaspace pre-tokenizer',
            (j) => j['pre_tokenizer'] = {
              'type': 'Metaspace',
              'replacement': _space,
            },
          ),
          (
            'an inverted Split',
            (j) => (j['pre_tokenizer'] as Map)['invert'] = true,
          ),
          (
            '<bos> at another id',
            (j) => ((j['model'] as Map)['vocab'] as Map)['<bos>'] = 9,
          ),
          (
            'an lstrip added token',
            (j) => ((j['added_tokens'] as List)[4] as Map)['lstrip'] = true,
          ),
        ]) {
      test('is not Gemma\'s shape with $name: no pass', () {
        final json = _toyHfJson();
        change(json);
        expect(UserDefinedPieces.fromHuggingFaceJson(json), isNull);
      });
    }
  });

  group('the dependency\'s own JSON format', () {
    test(
      'reads the piece types it keeps and matches the same pieces',
      () async {
        final fromModel = await loadEmbeddingTokenizer(
          await _write('toy.model', _toyModel()),
        );
        final tokenizer = await loadEmbeddingTokenizer(
          await _write('library.json', utf8.encode(fromModel.toJson())),
        );
        expectSentencePieceIds(tokenizer);
      },
    );
  });
}
