// What the language model is asked when the user asks a question: the
// retrieved items as numbered sources, which the answer cites as [1], [2].
// Pure Dart; unit-tested in test/knowledge_prompt_test.dart.

import 'knowledge_item.dart';

/// The system instruction of every answer.
const kKnowledgeSystemInstruction =
    'You answer questions about the user\'s own notes, voice memos and '
    'photos. Use only the numbered sources in the message. After each fact, '
    'cite its source number in square brackets, like [1]. If the sources do '
    'not contain the answer, say that you could not find it. Answer in at '
    'most three short sentences.';

/// A retrieved item as the language model gets it: numbered from 1, in rank
/// order. A photo is [attached] when its image is sent with the question.
typedef KnowledgeSource = ({int number, KnowledgeItem item, bool attached});

/// The user turn: the sources, then the question.
String buildKnowledgePrompt(String question, List<KnowledgeSource> sources) {
  final lines = <String>['Sources:'];
  for (final source in sources) {
    final item = source.item;
    final n = source.number;
    lines.add(switch (item) {
      NoteItem(:final title, :final body) => '[$n] Note "$title": $body',
      MemoItem(:final transcript?) =>
        '[$n] Voice memo, transcript: "${transcript.trim()}"',
      MemoItem() => '[$n] Voice memo (no transcript available)',
      PhotoItem() when source.attached => '[$n] Photo: the attached image',
      PhotoItem() => '[$n] Photo (not shown to you)',
    });
  }
  lines
    ..add('')
    ..add('Question: ${question.trim()}');
  return lines.join('\n');
}

/// The source numbers [answer] cites, in the order it first cites them.
/// Reads `[1]`, `[1, 3]` and `[1][2]`.
List<int> citedSources(String answer) {
  final cited = <int>[];
  for (final group in RegExp(r'\[([\d,\s]+)\]').allMatches(answer)) {
    for (final number in group.group(1)!.split(',')) {
      final n = int.tryParse(number.trim());
      if (n != null && !cited.contains(n)) cited.add(n);
    }
  }
  return cited;
}
