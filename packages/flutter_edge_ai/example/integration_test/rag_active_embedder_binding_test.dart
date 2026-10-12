// RAG on device (R5 §10.1, D20): a location opened through
// FlutterEdgeAiActiveEmbedder.forActiveModel is bound to the installed model's
// files; after another model of the SAME dimension is installed, its derived
// id differs and the location refuses it — the id is the only guard between
// two 768-dimensional embedding spaces. Plus: an explicit catalog id still
// works as before, and declaring images on a text-only model fails at open,
// before any store is created.
//
// Models: the ungated Gecko 110M seq64 and seq256 (768D both), downloaded at
// their pinned revision. The EmbeddingGemma 1 vs 2 leg of R5 §10.1 joins this
// suite with the EmbeddingGemma 2 engine (PR D).
//
// No setUp (Rule 6b): each test body calls the guarded helpers it needs.
//
// Run: flutter test integration_test/rag_active_embedder_binding_test.dart -d macos

import 'dart:io';

import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_example/models/embedding_model.dart' as catalog;
import 'package:flutter_edge_ai_example/services/embedding_catalog_install.dart';
import 'package:flutter_edge_ai_rag/flutter_edge_ai_rag.dart';
import 'package:flutter_edge_ai_sqlite/flutter_edge_ai_sqlite.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:path_provider/path_provider.dart';

import 'inference_test_helpers.dart' show registerTestEngines;

const _namespace = 'rag-binding-test';

const _documents = {
  'grid': 'Renewable energy is reshaping the global power grid.',
  'pizza': 'A classic margherita pizza needs only flour, tomato, and basil.',
  'markets': 'Stock markets reacted sharply to the central bank announcement.',
};

Future<void> _install(catalog.EmbeddingModel model) async {
  await registerTestEngines();
  await withCatalogEmbedderSources(
    FlutterEdgeAi.installEmbedder(),
    model,
  ).install();
}

Future<String> _freshLocation(String name) async {
  final dir = await getTemporaryDirectory();
  final path =
      '${dir.path}/${name}_${DateTime.now().microsecondsSinceEpoch}.db';
  addTearDown(() async {
    final file = File(path);
    if (await file.exists()) await file.delete();
  });
  return path;
}

final _rag = FlutterEdgeAiRag(providers: const [SqliteVectorStoreProvider()]);

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
    'forActiveModel binds the location to the installed files; another '
    'model of the same dimension is refused',
    (_) async {
      final location = await _freshLocation('rag_binding');
      final spec = VectorStoreSpec(
        providerId: SqliteVectorStoreProvider.providerId,
        location: location,
      );

      await _install(catalog.EmbeddingModel.gecko64);
      final first = FlutterEdgeAiActiveEmbedder.forActiveModel(
        namespace: _namespace,
      );
      final index = await _rag.open(spec: spec, embedder: first);
      final boundId = index.embeddingProfile.id;
      expect(
        boundId,
        '$_namespace@${catalog.EmbeddingModel.gecko64.filename}'
        '+${catalog.EmbeddingModel.gecko64.tokenizerFilename}',
      );
      expect(index.embeddingProfile.dimension, 768);
      for (final MapEntry(key: id, value: text) in _documents.entries) {
        await index.add(id: id, document: EmbeddingInput.text(text));
      }
      final hits = await index.search(
        query: EmbeddingInput.text('pizza recipe'),
        topK: 1,
      );
      expect(hits.single.id, 'pizza');
      await index.dispose();

      await _install(catalog.EmbeddingModel.gecko256);
      final second = FlutterEdgeAiActiveEmbedder.forActiveModel(
        namespace: _namespace,
      );
      final secondProfile = await second.resolveProfile();
      expect(secondProfile.dimension, 768);
      expect(secondProfile.id, isNot(boundId));
      await expectLater(
        _rag.open(spec: spec, embedder: second),
        throwsA(
          isA<StateError>().having(
            (error) => error.message,
            'message',
            allOf(contains('is bound to'), contains(boundId)),
          ),
        ),
      );
      await expectLater(
        first.embedDocument(EmbeddingInput.text('late')),
        throwsStateError,
        reason: 'the first adapter is pinned to the model it was made for',
      );

      // The explicit catalog id keeps working as before.
      final explicit = await _rag.open(
        spec: VectorStoreSpec(
          providerId: SqliteVectorStoreProvider.providerId,
          location: await _freshLocation('rag_binding_explicit'),
        ),
        embedder: FlutterEdgeAiActiveEmbedder(
          profileId: catalog.EmbeddingModel.gecko256.ragProfileId,
        ),
      );
      expect(
        explicit.embeddingProfile.id,
        catalog.EmbeddingModel.gecko256.ragProfileId,
      );
      await explicit.add(
        id: 'grid',
        document: EmbeddingInput.text(_documents['grid']!),
      );
      expect((await explicit.stats()).documentCount, 1);
      await explicit.dispose();
    },
    timeout: const Timeout(Duration(minutes: 15)),
  );

  testWidgets(
    'declaring images on a text-only model fails at open, before any store',
    (_) async {
      await _install(catalog.EmbeddingModel.gecko64);
      final location = await _freshLocation('rag_binding_images');

      await expectLater(
        _rag.open(
          spec: VectorStoreSpec(
            providerId: SqliteVectorStoreProvider.providerId,
            location: location,
            documents: EmbeddingModality.image,
          ),
          embedder: FlutterEdgeAiActiveEmbedder(
            profileId: 'gecko-images-v1',
            supportImage: true,
          ),
        ),
        throwsA(isA<EmbeddingLoadException>()),
      );
      expect(
        File(location).existsSync(),
        isFalse,
        reason: 'the model refused before the store was created',
      );
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
