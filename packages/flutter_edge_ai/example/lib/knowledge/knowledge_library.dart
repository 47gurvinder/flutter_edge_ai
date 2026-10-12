import 'dart:convert';

import '../image_search/photo_store.dart' show assetPhotoReference;
import 'knowledge_item.dart';
import 'knowledge_seed.dart';

/// What indexing one item cost: embedding it, and transcribing it first
/// (memos only).
typedef IndexCost = ({Duration embed, Duration? transcribe});

/// The knowledge base's items — the samples, then what the user added — and
/// which of them are in the index of each embedding space.
///
/// Saved as JSON: the user's items whole, the samples only as their
/// transcripts (they ship with the app), and the index costs per embedding
/// space ([indexed], keyed by the text profile id), so a model switch never
/// shows another space's items as indexed.
final class KnowledgeLibrary {
  KnowledgeLibrary._(this._items, this._indexed);

  /// The samples alone.
  factory KnowledgeLibrary.seeded() => KnowledgeLibrary._(_seedItems(), {});

  /// The samples plus what [json] saved. A corrupt save is reported, not
  /// dropped: [FormatException].
  factory KnowledgeLibrary.decode(String? json) {
    final library = KnowledgeLibrary.seeded();
    if (json == null || json.isEmpty) return library;
    final saved = jsonDecode(json);
    if (saved is! Map<String, Object?>) {
      throw const FormatException(
        'the saved knowledge library is not an object',
      );
    }
    final transcripts = (saved['transcripts'] as Map?)?.cast<String, String>();
    for (final (i, item) in library._items.indexed) {
      if (item case MemoItem(:final id) when transcripts?[id] != null) {
        library._items[i] = item.withTranscript(transcripts![id]!);
      }
    }
    for (final json in (saved['items'] as List?) ?? const []) {
      library._items.add(
        KnowledgeItem.fromJson((json as Map).cast<String, Object?>()),
      );
    }
    final indexed = (saved['indexed'] as Map?) ?? const {};
    for (final MapEntry(key: space, value: costs) in indexed.entries) {
      library._indexed[space as String] = {
        for (final MapEntry(key: id, value: cost) in (costs as Map).entries)
          id as String: (
            embed: Duration(microseconds: (cost as Map)['e'] as int),
            transcribe: switch (cost['t']) {
              final int us => Duration(microseconds: us),
              _ => null,
            },
          ),
      };
    }
    return library;
  }

  final List<KnowledgeItem> _items;
  final Map<String, Map<String, IndexCost>> _indexed;

  List<KnowledgeItem> get items => List.unmodifiable(_items);

  KnowledgeItem? byId(String id) {
    for (final item in _items) {
      if (item.id == id) return item;
    }
    return null;
  }

  /// What indexing each item cost in [space], for the items in its index.
  Map<String, IndexCost> indexedIn(String space) =>
      Map.unmodifiable(_indexed[space] ?? const {});

  void add(KnowledgeItem item) {
    if (byId(item.id) != null) {
      throw StateError('the library already holds ${item.id}');
    }
    _items.add(item);
  }

  void replace(KnowledgeItem item) {
    final i = _items.indexWhere((existing) => existing.id == item.id);
    if (i < 0) throw StateError('the library does not hold ${item.id}');
    _items[i] = item;
  }

  void markIndexed(String space, String id, IndexCost cost) =>
      (_indexed[space] ??= {})[id] = cost;

  /// Forgets what is indexed in [space] — of [kind] only, when given.
  void forgetIndexed(String space, {KnowledgeKind? kind}) {
    final costs = _indexed[space];
    if (costs == null) return;
    if (kind == null) {
      costs.clear();
    } else {
      costs.removeWhere((id, _) => byId(id)?.kind == kind);
    }
  }

  String encode() => jsonEncode({
    'version': 1,
    'items': [
      for (final item in _items)
        if (!item.seeded) item.toJson(),
    ],
    'transcripts': {
      for (final item in _items)
        if (item case MemoItem(seeded: true, :final id, :final transcript?))
          id: transcript,
    },
    'indexed': {
      for (final MapEntry(key: space, value: costs) in _indexed.entries)
        space: {
          for (final MapEntry(key: id, value: cost) in costs.entries)
            id: {
              'e': cost.embed.inMicroseconds,
              't': ?cost.transcribe?.inMicroseconds,
            },
        },
    },
  });
}

List<KnowledgeItem> _seedItems() => [
  for (final note in kSeedNotes)
    NoteItem(id: note.id, title: note.title, body: note.body, seeded: true),
  for (final photo in kSeedPhotos)
    PhotoItem(
      id: photo.id,
      title: photo.title,
      reference: assetPhotoReference(photo.asset),
      seeded: true,
    ),
  for (final memo in kSeedMemos)
    MemoItem(
      id: memo.id,
      title: memo.title,
      // The same `asset:` reference format as a photo's.
      reference: assetPhotoReference(memo.asset),
      duration: memo.duration,
      seeded: true,
    ),
];
