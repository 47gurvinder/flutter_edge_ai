// The Knowledge screen's retrieval with the real EmbeddingGemma 2 (740M):
// the sample notes, photos and voice memos go into three SQLite indexes
// through the screen's own KnowledgeIndex, and every text query the Search
// tab suggests must rank its item first ACROSS the three kinds
// (knowledge_ranking.dart: lift over each kind's median). Prints `[knowledge]`
// lines: per-item embedding times, per-query search times, the ranked lifts.
//
// No setUp/setUpAll (Rule 6b). Runs where the 740M runs natively:
//   flutter test integration_test/knowledge_retrieval_test.dart -d <device>
// Optional: --dart-define=KNOWLEDGE_EMBEDDER_FILE=<a local copy of the
// bundle> (checked against the catalog's SHA-256), else it is downloaded
// (485 MB, ungated).

import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_example/gemma_bootstrap.dart';
import 'package:flutter_edge_ai_example/image_search/photo_preparation.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_app_dependencies.dart'
    show knowledgeProfileId;
import 'package:flutter_edge_ai_example/knowledge/knowledge_dependencies.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_index.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_item.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_library.dart';
import 'package:flutter_edge_ai_example/knowledge/knowledge_widgets.dart'
    show kKnowledgeSuggestions;
import 'package:flutter_edge_ai_example/models/embedding_model.dart' as catalog;
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_edge_ai_sqlite/flutter_edge_ai_sqlite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

const _file = String.fromEnvironment('KNOWLEDGE_EMBEDDER_FILE');
const _model = catalog.EmbeddingModel.embeddingGemma2TextVisionAudio740m;

/// The item each suggested query must rank first.
const _expectedFirst = {
  'my flight is delayed': 'memo-flight',
  'the airport taxi receipt': 'photo-0412',
  'flickering lights': 'memo-kitchen',
  'market share chart': 'photo-0291',
  'fresh basil': 'memo-groceries',
  'revenue by quarter': 'note-q3',
  'a cooking class': 'memo-offsite',
  // Russian: one space across languages.
  'чек за такси': 'photo-0412',
};

void _log(String line) {
  // ignore: avoid_print
  print('[knowledge] $line');
}

Future<Uint8List> _asset(String key) async {
  final data = await rootBundle.load(key);
  return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}

Future<void> _ensureModel() async {
  await bootstrapGemma();
  final builder = FlutterEdgeAi.installEmbedder();
  if (_file.isNotEmpty) {
    final digest = await sha256.bind(File(_file).openRead()).first;
    expect('$digest', _model.sha256, reason: '$_file is not the pinned bundle');
    builder.modelFromFile(_file, filename: _model.filename);
  } else {
    builder.modelFromNetwork(_model.url, filename: _model.filename);
  }
  await builder.install();
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('every suggested query ranks its item first across notes, '
      'photos and voice memos', (tester) async {
    expect(_expectedFirst.keys, kKnowledgeSuggestions);
    await _ensureModel();
    final directory = await Directory(
      '${(await getTemporaryDirectory()).path}/knowledge_retrieval_'
      '${DateTime.now().microsecondsSinceEpoch}',
    ).create(recursive: true);
    final info = KnowledgeEmbedderInfo(
      name: _model.displayName,
      textProfileId: _model.ragProfileId,
      kinds: KnowledgeKind.values.toSet(),
    );
    final watch = Stopwatch()..start();
    final index = await KnowledgeIndex.open(
      kinds: info.kinds,
      embedderFor: (kind) => FlutterEdgeAiActiveEmbedder(
        profileId: knowledgeProfileId(kind, info),
        supportImage: true,
        supportAudio: true,
      ),
      openIndex: (kind, embedder) => exampleRag.open(
        spec: VectorStoreSpec(
          providerId: SqliteVectorStoreProvider.providerId,
          location: '${directory.path}/${kind.name}.db',
          documents: kind.modality,
        ),
        embedder: embedder,
      ),
    );
    _log('opened (model loaded) in ${watch.elapsedMilliseconds} ms');
    try {
      final times = <KnowledgeKind, List<int>>{};
      for (final item in KnowledgeLibrary.seeded().items) {
        final took = switch (item) {
          NoteItem() => await index.addNote(item),
          PhotoItem(:final reference) => await index.addPhoto(
            item,
            (await preparePhotoForEmbedding(
              await _asset(reference.substring('asset:'.length)),
            )).bytes,
          ),
          MemoItem(:final reference) => await index.addMemo(
            item,
            await _asset(reference.substring('asset:'.length)),
          ),
        };
        (times[item.kind] ??= []).add(took.inMilliseconds);
      }
      for (final MapEntry(key: kind, value: ms) in times.entries) {
        final sorted = [...ms]..sort();
        _log(
          '${kind.plural}: ${ms.length}, embed median '
          '${sorted[sorted.length ~/ 2]} ms, first ${ms.first} ms',
        );
      }
      final misses = <String>[];
      for (final MapEntry(key: query, value: expected)
          in _expectedFirst.entries) {
        watch.reset();
        final hits = await index.search(query);
        final ms = watch.elapsedMilliseconds;
        _log(
          '"$query" ($ms ms): '
          '${[for (final h in hits) '${h.id} ${h.lift.toStringAsFixed(3)}'].join(', ')}',
        );
        if (hits.isEmpty || hits.first.id != expected) {
          misses.add('"$query" → ${hits.firstOrNull?.id}, not $expected');
        }
      }
      expect(misses, isEmpty);
    } finally {
      await index.dispose();
    }
  });
}
