import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import 'knowledge_item.dart';
import 'knowledge_prompt.dart';
import 'knowledge_ranking.dart';
import 'knowledge_theme.dart';
import 'knowledge_view_model.dart';

/// The model lists the screen sends the user to.
enum KnowledgeModelList { embedder, stt, llm }

typedef OpenModelList = Future<void> Function(KnowledgeModelList list);

/// Text queries the Search tab suggests. On the sample set each finds its
/// item first: a voice memo, a photo, a memo, a chart, a memo, a note (with
/// its chart second) and a memo.
const kKnowledgeSuggestions = [
  'my flight is delayed',
  'the airport taxi receipt',
  'flickering lights',
  'market share chart',
  'fresh basil',
  'revenue by quarter',
  'a cooking class',
];

/// Questions the Ask tab suggests.
const kKnowledgeQuestions = [
  'What was the total on my airport taxi receipt, and will finance pay it '
      'back?',
  'What is going on with the kitchen?',
  'Who do I have to call on Sunday?',
];

// ── Frame ───────────────────────────────────────────────────────────────────

class KnowledgeTitle extends StatelessWidget {
  const KnowledgeTitle({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text('Knowledge'),
        Text(
          'EmbeddingGemma 2 + Gemma 4',
          style: theme.textTheme.labelMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class KnowledgeLoading extends StatelessWidget {
  const KnowledgeLoading({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 36,
            height: 36,
            child: CircularProgressIndicator(strokeWidth: 3),
          ),
          const SizedBox(height: 16),
          Text('Loading EmbeddingGemma 2…', style: theme.textTheme.titleSmall),
          const SizedBox(height: 4),
          Text(
            'Opening the notes, photos and memos indexes',
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}

/// No usable embedder (or the indexes did not open): what is wrong, and the
/// way to the model list.
class KnowledgeSetupProblem extends StatelessWidget {
  const KnowledgeSetupProblem({
    super.key,
    required this.vm,
    required this.onOpenList,
  });

  final KnowledgeViewModel vm;
  final OpenModelList onOpenList;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        Icon(Icons.hub_outlined, size: 56, color: theme.colorScheme.primary),
        const SizedBox(height: 16),
        Text(
          'Knowledge needs EmbeddingGemma 2',
          style: theme.textTheme.titleLarge,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          vm.error ?? 'The embedding model did not load.',
          key: const Key('knowledge-setup-problem'),
          style: theme.textTheme.bodyMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 20),
        FilledButton.icon(
          key: const Key('knowledge-open-embedder-list'),
          onPressed: () => onOpenList(KnowledgeModelList.embedder),
          icon: const Icon(Icons.download_outlined),
          label: const Text('Open the Embedding Models list'),
        ),
        const SizedBox(height: 8),
        TextButton(onPressed: vm.open, child: const Text('Try again')),
      ],
    );
  }
}

/// A failure, until dismissed.
class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    final error = vm.error;
    if (error == null) return const SizedBox.shrink();
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Material(
        key: const Key('knowledge-error'),
        color: scheme.errorContainer,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(14, 10, 4, 10),
          child: Row(
            children: [
              Icon(Icons.error_outline, color: scheme.onErrorContainer),
              const SizedBox(width: 10),
              Expanded(
                child: SelectableText(
                  error,
                  style: TextStyle(color: scheme.onErrorContainer),
                ),
              ),
              IconButton(
                tooltip: 'Dismiss',
                onPressed: vm.clearError,
                icon: Icon(Icons.close, color: scheme.onErrorContainer),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A heading over a section.
class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.title, {this.trailing});

  final String title;
  final Widget? trailing;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(4, 20, 4, 8),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: theme.textTheme.titleSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
                letterSpacing: 0.3,
              ),
            ),
          ),
          ?trailing,
        ],
      ),
    );
  }
}

// ── Library ─────────────────────────────────────────────────────────────────

class LibraryTab extends StatelessWidget {
  const LibraryTab({super.key, required this.vm, required this.onOpenList});

  final KnowledgeViewModel vm;
  final OpenModelList onOpenList;

  @override
  Widget build(BuildContext context) {
    final setup = vm.setup!;
    return ListView(
      key: const Key('knowledge-library'),
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      children: [
        _ErrorBanner(vm: vm),
        _ModelsCard(vm: vm, onOpenList: onOpenList),
        const SizedBox(height: 12),
        _IndexCard(vm: vm),
        if (vm.recording || vm.transcribingMemo) ...[
          const SizedBox(height: 12),
          _RecordingBanner(vm: vm),
        ],
        for (final kind in KnowledgeKind.values)
          if (kind != KnowledgeKind.memo || setup.audioSupported) ...[
            _SectionHeader(
              '${kind.plural.toUpperCase()} · ${vm.itemsOf(kind).length}',
              trailing: _AddButton(vm: vm, kind: kind),
            ),
            if (!vm.indexes(kind))
              _Notice(
                icon: Icons.info_outline,
                text: switch (kind) {
                  KnowledgeKind.photo =>
                    'Photos need EmbeddingGemma 2 with a vision encoder (440M '
                        'or 740M).',
                  _ =>
                    'Voice memos need EmbeddingGemma 2 Text + Vision + Audio '
                        '(740M).',
                },
                action: 'Choose',
                onAction: () => onOpenList(KnowledgeModelList.embedder),
              ),
            if (kind == KnowledgeKind.memo && setup.stt == null)
              _Notice(
                icon: Icons.mic_none,
                text:
                    'Install a speech-to-text model (Moonshine Tiny) for memo '
                    'transcripts.',
                action: 'Install',
                onAction: () => onOpenList(KnowledgeModelList.stt),
              ),
            switch (kind) {
              KnowledgeKind.note => Column(
                children: [
                  for (final note in vm.itemsOf(kind).cast<NoteItem>())
                    _NoteTile(vm: vm, note: note),
                ],
              ),
              KnowledgeKind.photo => _PhotoGrid(
                vm: vm,
                photos: vm.itemsOf(kind).cast<PhotoItem>(),
              ),
              KnowledgeKind.memo => Column(
                children: [
                  for (final memo in vm.itemsOf(kind).cast<MemoItem>())
                    _MemoTile(vm: vm, memo: memo),
                ],
              ),
            },
          ],
      ],
    );
  }
}

/// The three models, what they run on, and the way to each model list.
class _ModelsCard extends StatelessWidget {
  const _ModelsCard({required this.vm, required this.onOpenList});

  final KnowledgeViewModel vm;
  final OpenModelList onOpenList;

  @override
  Widget build(BuildContext context) {
    final setup = vm.setup!;
    final embedder = setup.embedder!;
    final llm = setup.llm;
    final stt = setup.stt;
    return Card(
      key: const Key('knowledge-models'),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Column(
          children: [
            _ModelRow(
              icon: Icons.hub_outlined,
              role: 'Embeddings',
              name: embedder.name,
              detail: [
                embedder.backend?.name ?? (kIsWeb ? 'webgpu' : 'cpu'),
                if (vm.openTook case final took?)
                  'opened in ${shortDuration(took)}',
              ].join(' · '),
              onTap: () => onOpenList(KnowledgeModelList.embedder),
            ),
            if (setup.audioSupported)
              _ModelRow(
                icon: Icons.mic_none,
                role: 'Speech to text',
                name: stt?.name,
                missing: 'Install Moonshine Tiny',
                onTap: () => onOpenList(KnowledgeModelList.stt),
              ),
            _ModelRow(
              icon: Icons.auto_awesome_outlined,
              role: 'Answers',
              name: llm?.name,
              missing: 'Install Gemma 4 E2B',
              detail: switch (vm.llmState) {
                LlmState.loading => 'loading…',
                LlmState.ready =>
                  '${vm.llmBackend} · loaded in ${shortDuration(vm.llmLoadTook!)}',
                LlmState.failed => 'failed to load',
                LlmState.idle => 'loads when needed',
                LlmState.missing => null,
              },
              busy: vm.llmState == LlmState.loading,
              onTap: () => onOpenList(KnowledgeModelList.llm),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModelRow extends StatelessWidget {
  const _ModelRow({
    required this.icon,
    required this.role,
    required this.name,
    required this.onTap,
    this.detail,
    this.missing,
    this.busy = false,
  });

  final IconData icon;
  final String role;
  final String? name;
  final String? detail;
  final String? missing;
  final bool busy;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final present = name != null;
    return ListTile(
      dense: true,
      visualDensity: VisualDensity.compact,
      onTap: onTap,
      leading: Icon(icon, color: present ? scheme.primary : scheme.error),
      title: Text(
        present ? name! : missing!,
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
        style: theme.textTheme.bodyMedium?.copyWith(
          fontWeight: FontWeight.w600,
          color: present ? null : scheme.error,
        ),
      ),
      subtitle: Text(
        [role, ?detail].join(' · '),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: busy
          ? const SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : Icon(
              present ? Icons.check_circle : Icons.chevron_right,
              size: 18,
              color: present ? scheme.primary : scheme.error,
            ),
    );
  }
}

/// How much is indexed, the run in progress, and the two index commands.
class _IndexCard extends StatelessWidget {
  const _IndexCard({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final indexable = [
      for (final item in vm.items)
        if (vm.indexes(item.kind)) item,
    ];
    final done = vm.indexedCount;
    final total = indexable.length;
    final current = vm.indexingId == null
        ? null
        : vm.items.where((i) => i.id == vm.indexingId).firstOrNull;
    return Card(
      key: const Key('knowledge-index-card'),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    '$done of $total indexed',
                    key: const Key('knowledge-indexed-count'),
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                if (done > 0 && !vm.busy)
                  TextButton(
                    key: const Key('knowledge-clear'),
                    onPressed: vm.clearIndex,
                    child: const Text('Clear'),
                  ),
              ],
            ),
            const SizedBox(height: 10),
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: TweenAnimationBuilder<double>(
                tween: Tween(end: total == 0 ? 0 : done / total),
                duration: const Duration(milliseconds: 350),
                curve: Curves.easeOut,
                builder: (context, value, _) =>
                    LinearProgressIndicator(value: value, minHeight: 8),
              ),
            ),
            const SizedBox(height: 10),
            Text(
              vm.indexing
                  ? 'Embedding ${current?.kind.label.toLowerCase() ?? 'item'}'
                        '${current == null ? '' : ' “${current.title}”'}…'
                  : vm.lastRunTook != null
                  ? 'Last run took ${shortDuration(vm.lastRunTook!)} on the '
                        'device. Every item: one EmbeddingGemma 2 vector.'
                  : 'Notes as text, photos as pixels, memos as sound — one '
                        'embedding space.',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            if (vm.pending.isNotEmpty && !vm.indexing) ...[
              const SizedBox(height: 12),
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  key: const Key('knowledge-index'),
                  onPressed: vm.busy ? null : vm.indexLibrary,
                  icon: const Icon(Icons.bolt),
                  label: Text(
                    'Index ${vm.pending.length} '
                    'item${vm.pending.length == 1 ? '' : 's'}',
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// The "+" of a section: a note, a photo, or a recorded memo.
class _AddButton extends StatelessWidget {
  const _AddButton({required this.vm, required this.kind});

  final KnowledgeViewModel vm;
  final KnowledgeKind kind;

  @override
  Widget build(BuildContext context) {
    final enabled = !vm.busy && vm.indexes(kind);
    return switch (kind) {
      KnowledgeKind.note => TextButton.icon(
        key: const Key('knowledge-add-note'),
        onPressed: enabled ? () => _addNote(context) : null,
        icon: const Icon(Icons.add, size: 18),
        label: const Text('Note'),
      ),
      KnowledgeKind.photo => TextButton.icon(
        key: const Key('knowledge-add-photo'),
        onPressed: enabled ? vm.addPhoto : null,
        icon: const Icon(Icons.add_photo_alternate_outlined, size: 18),
        label: const Text('Photo'),
      ),
      KnowledgeKind.memo => TextButton.icon(
        key: const Key('knowledge-record'),
        onPressed: enabled ? vm.startRecording : null,
        icon: const Icon(Icons.fiber_manual_record, size: 18),
        label: const Text('Record'),
      ),
    };
  }

  Future<void> _addNote(BuildContext context) async {
    final title = TextEditingController();
    final body = TextEditingController();
    final saved = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (context) => Padding(
        padding: EdgeInsets.fromLTRB(
          20,
          0,
          20,
          20 + MediaQuery.viewInsetsOf(context).bottom,
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text('New note', style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 12),
            TextField(
              key: const Key('knowledge-note-title'),
              controller: title,
              decoration: const InputDecoration(labelText: 'Title'),
              textInputAction: TextInputAction.next,
            ),
            const SizedBox(height: 8),
            TextField(
              key: const Key('knowledge-note-body'),
              controller: body,
              decoration: const InputDecoration(
                labelText: 'Note (markdown)',
                alignLabelWithHint: true,
              ),
              minLines: 3,
              maxLines: 8,
            ),
            const SizedBox(height: 16),
            FilledButton(
              key: const Key('knowledge-note-save'),
              onPressed: () => Navigator.pop(context, true),
              child: const Text('Save and index'),
            ),
          ],
        ),
      ),
    );
    final text = body.text;
    final heading = title.text;
    title.dispose();
    body.dispose();
    if (saved == true && text.trim().isNotEmpty) {
      await vm.addNote(title: heading, body: text);
    }
  }
}

class _Notice extends StatelessWidget {
  const _Notice({
    required this.icon,
    required this.text,
    required this.action,
    required this.onAction,
  });

  final IconData icon;
  final String text;
  final String action;
  final VoidCallback onAction;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Material(
        color: scheme.tertiaryContainer,
        borderRadius: BorderRadius.circular(14),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 6, 4, 6),
          child: Row(
            children: [
              Icon(icon, size: 18, color: scheme.onTertiaryContainer),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  text,
                  style: TextStyle(color: scheme.onTertiaryContainer),
                ),
              ),
              TextButton(onPressed: onAction, child: Text(action)),
            ],
          ),
        ),
      ),
    );
  }
}

/// An item's index state: a spinner while it is embedded, the time it took
/// once it is, the reason when it failed.
class _ItemStatus extends StatelessWidget {
  const _ItemStatus({required this.vm, required this.item});

  final KnowledgeViewModel vm;
  final KnowledgeItem item;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.labelSmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
    );
    final Widget child;
    if (vm.indexingId == item.id) {
      child = const SizedBox(
        width: 16,
        height: 16,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    } else if (vm.costOf(item.id) case final cost?) {
      child = Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.check_circle, size: 14, color: theme.colorScheme.primary),
          const SizedBox(width: 4),
          Text(
            [
              if (cost.transcribe case final stt?) 'STT ${shortDuration(stt)}',
              shortDuration(cost.embed),
            ].join(' · '),
            style: muted,
          ),
        ],
      );
    } else if (vm.failureOf(item.id) case final failure?) {
      child = Tooltip(
        message: failure,
        child: Icon(
          Icons.error_outline,
          size: 16,
          color: theme.colorScheme.error,
        ),
      );
    } else if (vm.isQueued(item.id)) {
      child = Icon(Icons.schedule, size: 14, color: muted?.color);
    } else {
      child = Text(vm.indexes(item.kind) ? 'not indexed' : '—', style: muted);
    }
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 250),
      child: KeyedSubtree(
        key: ValueKey('${item.id}-${child.runtimeType}-${vm.costOf(item.id)}'),
        child: child,
      ),
    );
  }
}

/// A round tile in a kind's color.
class _KindBadge extends StatelessWidget {
  const _KindBadge({required this.kind, this.size = 40});

  final KnowledgeKind kind;
  final double size;

  @override
  Widget build(BuildContext context) {
    final color = kindColor(kind, Theme.of(context).brightness);
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        borderRadius: BorderRadius.circular(size * 0.32),
      ),
      child: Icon(kindIcon(kind), color: color, size: size * 0.5),
    );
  }
}

class _NoteTile extends StatelessWidget {
  const _NoteTile({required this.vm, required this.note});

  final KnowledgeViewModel vm;
  final NoteItem note;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Card(
        key: Key('knowledge-item-${note.id}'),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const _KindBadge(kind: KnowledgeKind.note, size: 36),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            note.title,
                            style: theme.textTheme.titleSmall,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        _ItemStatus(vm: vm, item: note),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      note.body,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PhotoGrid extends StatelessWidget {
  const _PhotoGrid({required this.vm, required this.photos});

  final KnowledgeViewModel vm;
  final List<PhotoItem> photos;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = math.max(3, (constraints.maxWidth / 140).floor());
        const gap = 8.0;
        final size = (constraints.maxWidth - gap * (columns - 1)) / columns;
        return Wrap(
          spacing: gap,
          runSpacing: gap,
          children: [
            for (final photo in photos)
              SizedBox(
                key: Key('knowledge-item-${photo.id}'),
                width: size,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Stack(
                      children: [
                        KnowledgeThumb(
                          image: vm.imageFor(photo.reference),
                          size: size,
                        ),
                        Positioned(
                          right: 6,
                          bottom: 6,
                          child: Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 6,
                              vertical: 3,
                            ),
                            decoration: BoxDecoration(
                              color: Theme.of(
                                context,
                              ).colorScheme.surface.withValues(alpha: 0.85),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: _ItemStatus(vm: vm, item: photo),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      photo.title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: Theme.of(context).textTheme.labelSmall,
                    ),
                  ],
                ),
              ),
          ],
        );
      },
    );
  }
}

class _MemoTile extends StatelessWidget {
  const _MemoTile({required this.vm, required this.memo});

  final KnowledgeViewModel vm;
  final MemoItem memo;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Card(
        key: Key('knowledge-item-${memo.id}'),
        child: Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              MemoPlayButton(vm: vm, memo: memo),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            '${memo.title} · ${clock(memo.duration)}',
                            style: theme.textTheme.titleSmall,
                          ),
                        ),
                        _ItemStatus(vm: vm, item: memo),
                      ],
                    ),
                    const SizedBox(height: 2),
                    AnimatedSwitcher(
                      duration: const Duration(milliseconds: 300),
                      child: Text(
                        memo.transcript == null
                            ? 'Transcribed when it is indexed'
                            : '“${memo.transcript}”',
                        key: ValueKey(memo.transcript),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                          fontStyle: memo.transcript == null
                              ? FontStyle.normal
                              : FontStyle.italic,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Recording (with its clock and Stop), then transcribing.
class _RecordingBanner extends StatelessWidget {
  const _RecordingBanner({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final since = vm.recordingSince;
    return Material(
      key: const Key('knowledge-recording'),
      color: scheme.errorContainer,
      borderRadius: BorderRadius.circular(16),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 8, 8),
        child: Row(
          children: [
            Icon(
              vm.recording ? Icons.fiber_manual_record : Icons.hearing,
              color: scheme.error,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: since == null
                  ? Text(
                      'Keeping the memo…',
                      style: TextStyle(color: scheme.onErrorContainer),
                    )
                  : StreamBuilder<int>(
                      stream: Stream.periodic(
                        const Duration(milliseconds: 250),
                        (i) => i,
                      ),
                      builder: (context, _) => Text(
                        'Recording ${clock(DateTime.now().difference(since))}'
                        ' / ${clock(kMaxMemo)}',
                        style: TextStyle(color: scheme.onErrorContainer),
                      ),
                    ),
            ),
            if (vm.recording)
              FilledButton(
                key: const Key('knowledge-stop-recording'),
                onPressed: vm.stopRecording,
                child: const Text('Stop'),
              ),
          ],
        ),
      ),
    );
  }
}

/// Plays a memo; a moving waveform while it plays.
class MemoPlayButton extends StatelessWidget {
  const MemoPlayButton({
    super.key,
    required this.vm,
    required this.memo,
    this.size = 40,
  });

  final KnowledgeViewModel vm;
  final MemoItem memo;
  final double size;

  @override
  Widget build(BuildContext context) {
    final color = kindColor(KnowledgeKind.memo, Theme.of(context).brightness);
    return ValueListenableBuilder<String?>(
      valueListenable: vm.playing,
      builder: (context, playing, _) {
        final isPlaying = playing == memo.id;
        return Material(
          color: isPlaying ? color : color.withValues(alpha: 0.16),
          shape: const CircleBorder(),
          child: InkWell(
            key: Key('knowledge-play-${memo.id}'),
            customBorder: const CircleBorder(),
            onTap: () => vm.togglePlay(memo),
            child: SizedBox(
              width: size,
              height: size,
              child: isPlaying
                  ? _Waveform(color: Theme.of(context).colorScheme.surface)
                  : Icon(Icons.play_arrow_rounded, color: color),
            ),
          ),
        );
      },
    );
  }
}

class _Waveform extends StatefulWidget {
  const _Waveform({required this.color});

  final Color color;

  @override
  State<_Waveform> createState() => _WaveformState();
}

class _WaveformState extends State<_Waveform>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 900),
  )..repeat();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) => Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          for (var i = 0; i < 4; i++)
            Container(
              width: 3,
              height:
                  6 +
                  10 *
                      (0.5 +
                          0.5 *
                              math.sin(
                                (_controller.value + i * 0.23) * 2 * math.pi,
                              )),
              margin: const EdgeInsets.symmetric(horizontal: 1.5),
              decoration: BoxDecoration(
                color: widget.color,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
        ],
      ),
    );
  }
}

/// A photo, decoded at the size it is shown.
class KnowledgeThumb extends StatelessWidget {
  const KnowledgeThumb({super.key, required this.image, required this.size});

  final ImageProvider image;
  final double size;

  @override
  Widget build(BuildContext context) {
    final px = (size * math.max(2.0, MediaQuery.devicePixelRatioOf(context)))
        .round();
    final scheme = Theme.of(context).colorScheme;
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Image(
        image: ResizeImage(
          image,
          width: px,
          height: px,
          policy: ResizeImagePolicy.fit,
        ),
        width: size,
        height: size,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stackTrace) => Container(
          width: size,
          height: size,
          color: scheme.surfaceContainerHighest,
          child: Icon(Icons.broken_image_outlined, color: scheme.outline),
        ),
      ),
    );
  }
}

// ── Search ──────────────────────────────────────────────────────────────────

class SearchTab extends StatelessWidget {
  const SearchTab({
    super.key,
    required this.vm,
    required this.controller,
    required this.onSearch,
  });

  final KnowledgeViewModel vm;
  final TextEditingController controller;
  final void Function(String text) onSearch;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final results = vm.results;
    final query = vm.resultsFor;
    return ListView(
      key: const Key('knowledge-search'),
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      children: [
        _ErrorBanner(vm: vm),
        SearchBar(
          key: const Key('knowledge-search-field'),
          controller: controller,
          hintText: 'Search notes, photos and memos',
          leading: const Padding(
            padding: EdgeInsets.only(left: 4),
            child: Icon(Icons.search),
          ),
          trailing: [
            if (vm.searching)
              const Padding(
                padding: EdgeInsets.all(12),
                child: SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              )
            else
              IconButton(
                key: const Key('knowledge-search-go'),
                tooltip: 'Search',
                onPressed: () => onSearch(controller.text),
                icon: const Icon(Icons.arrow_forward),
              ),
          ],
          elevation: const WidgetStatePropertyAll(0),
          backgroundColor: WidgetStatePropertyAll(
            theme.colorScheme.surfaceContainerHighest,
          ),
          textInputAction: TextInputAction.search,
          onSubmitted: onSearch,
        ),
        const SizedBox(height: 12),
        _Suggestions(
          keyPrefix: 'knowledge-suggestion',
          suggestions: kKnowledgeSuggestions,
          onTap: onSearch,
        ),
        if (query != null) ...[
          const SizedBox(height: 8),
          _SectionHeader(
            results.isEmpty
                ? 'NO STRONG MATCH FOR “$query”'
                : '${results.length} RESULT${results.length == 1 ? '' : 'S'}'
                      '${vm.searchTook == null ? '' : ' · ${shortDuration(vm.searchTook!)}'}',
            trailing: Text(
              'text · images · audio',
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
          for (final (i, result) in results.indexed)
            _Appear(
              key: ValueKey('$query-${result.item.id}'),
              delay: Duration(milliseconds: 70 * i),
              child: Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: _ResultCard(
                  key: Key('knowledge-result-$i'),
                  vm: vm,
                  result: result,
                  query: query,
                  rank: i + 1,
                ),
              ),
            ),
          if (results.isEmpty)
            Text(
              'Nothing stands out from the rest of its kind. Try one of the '
              'suggestions.',
              style: theme.textTheme.bodyMedium,
            ),
        ] else if (vm.indexedCount == 0)
          const Padding(
            padding: EdgeInsets.only(top: 24),
            child: _EmptyHint(
              icon: Icons.collections_bookmark_outlined,
              text: 'Index the library first (Library tab).',
            ),
          )
        else
          const Padding(
            padding: EdgeInsets.only(top: 24),
            child: _EmptyHint(
              icon: Icons.travel_explore,
              text:
                  'One query, three kinds of memory: notes by their words, '
                  'photos by their pixels, voice memos by their sound.',
            ),
          ),
      ],
    );
  }
}

class _EmptyHint extends StatelessWidget {
  const _EmptyHint({required this.icon, required this.text});

  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      children: [
        Icon(icon, size: 40, color: theme.colorScheme.outline),
        const SizedBox(height: 8),
        Text(
          text,
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class _Suggestions extends StatelessWidget {
  const _Suggestions({
    required this.keyPrefix,
    required this.suggestions,
    required this.onTap,
  });

  final String keyPrefix;
  final List<String> suggestions;
  final void Function(String text) onTap;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 40,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: suggestions.length,
        separatorBuilder: (_, _) => const SizedBox(width: 8),
        itemBuilder: (context, i) => ActionChip(
          key: Key('$keyPrefix-$i'),
          avatar: const Icon(Icons.north_east, size: 16),
          label: Text(suggestions[i]),
          onPressed: () => onTap(suggestions[i]),
        ),
      ),
    );
  }
}

/// Fades and slides its child in, after [delay].
class _Appear extends StatefulWidget {
  const _Appear({super.key, required this.child, this.delay = Duration.zero});

  final Widget child;
  final Duration delay;

  @override
  State<_Appear> createState() => _AppearState();
}

class _AppearState extends State<_Appear> with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 320),
  );
  Timer? _start;

  @override
  void initState() {
    super.initState();
    _start = Timer(widget.delay, _controller.forward);
  }

  @override
  void dispose() {
    _start?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final curve = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    return FadeTransition(
      opacity: curve,
      child: SlideTransition(
        position: Tween(
          begin: const Offset(0, 0.08),
          end: Offset.zero,
        ).animate(curve),
        child: widget.child,
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({
    super.key,
    required this.vm,
    required this.result,
    required this.query,
    required this.rank,
  });

  final KnowledgeViewModel vm;
  final KnowledgeResult result;
  final String query;
  final int rank;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final item = result.item;
    final hit = result.hit;
    final color = kindColor(item.kind, theme.brightness);
    final leading = switch (item) {
      PhotoItem(:final reference) => KnowledgeThumb(
        image: vm.imageFor(reference),
        size: 72,
      ),
      MemoItem() => MemoPlayButton(vm: vm, memo: item, size: 52),
      NoteItem() => const _KindBadge(kind: KnowledgeKind.note, size: 52),
    };
    final snippet = switch (item) {
      NoteItem(:final body) => body,
      MemoItem(:final transcript?) => '“$transcript”',
      MemoItem() => 'No transcript yet',
      PhotoItem() => 'Found by what is in the picture — no caption, no OCR',
    };
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                leading,
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(kindIcon(item.kind), size: 14, color: color),
                          const SizedBox(width: 4),
                          Text(
                            [
                              item.kind.label.toUpperCase(),
                              if (item case MemoItem(:final duration))
                                clock(duration),
                            ].join(' · '),
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: color,
                              fontWeight: FontWeight.w700,
                              letterSpacing: 0.6,
                            ),
                          ),
                          const Spacer(),
                          Text(
                            '#$rank',
                            style: theme.textTheme.labelSmall?.copyWith(
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        item.title,
                        style: theme.textTheme.titleSmall,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 4),
                      _Highlighted(
                        text: snippet,
                        query: item is PhotoItem ? '' : query,
                        color: color,
                        italic: item is MemoItem,
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            _RelevanceBar(hit: hit, color: color),
          ],
        ),
      ),
    );
  }
}

class _Highlighted extends StatelessWidget {
  const _Highlighted({
    required this.text,
    required this.query,
    required this.color,
    this.italic = false,
  });

  final String text;
  final String query;
  final Color color;
  final bool italic;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final base = theme.textTheme.bodySmall?.copyWith(
      color: theme.colorScheme.onSurfaceVariant,
      fontStyle: italic ? FontStyle.italic : null,
      height: 1.35,
    );
    return Text.rich(
      TextSpan(
        style: base,
        children: [
          for (final part in highlight(text, query))
            TextSpan(
              text: part.text,
              style: part.match
                  ? TextStyle(
                      color: theme.colorScheme.onSurface,
                      fontWeight: FontWeight.w700,
                      backgroundColor: color.withValues(alpha: 0.22),
                    )
                  : null,
            ),
        ],
      ),
      maxLines: 3,
      overflow: TextOverflow.ellipsis,
    );
  }
}

/// How far a hit stands out from the rest of its kind (its lift, see
/// knowledge_ranking.dart), with the raw cosine beside it.
class _RelevanceBar extends StatelessWidget {
  const _RelevanceBar({required this.hit, required this.color});

  final RankedHit hit;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: hit.strength),
              duration: const Duration(milliseconds: 700),
              curve: Curves.easeOutCubic,
              builder: (context, value, _) => LinearProgressIndicator(
                value: value,
                minHeight: 6,
                color: color,
                backgroundColor: color.withValues(alpha: 0.14),
              ),
            ),
          ),
        ),
        const SizedBox(width: 10),
        Text(
          'cos ${hit.similarity.toStringAsFixed(2)}',
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
            fontFeatures: const [FontFeature.tabularFigures()],
          ),
        ),
      ],
    );
  }
}

// ── Ask ─────────────────────────────────────────────────────────────────────

class AskTab extends StatelessWidget {
  const AskTab({
    super.key,
    required this.vm,
    required this.controller,
    required this.onAsk,
    required this.onOpenList,
  });

  final KnowledgeViewModel vm;
  final TextEditingController controller;
  final void Function(String text) onAsk;
  final OpenModelList onOpenList;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final llm = vm.setup?.llm;
    return ListView(
      key: const Key('knowledge-ask'),
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
      children: [
        _ErrorBanner(vm: vm),
        if (llm == null)
          _Notice(
            icon: Icons.auto_awesome_outlined,
            text:
                'Answers need a language model: install Gemma 4 E2B from the '
                'Inference Models list.',
            action: 'Install',
            onAction: () => onOpenList(KnowledgeModelList.llm),
          ),
        TextField(
          key: const Key('knowledge-ask-field'),
          controller: controller,
          enabled: llm != null,
          minLines: 1,
          maxLines: 4,
          textInputAction: TextInputAction.send,
          onSubmitted: onAsk,
          decoration: InputDecoration(
            hintText: 'Ask about your notes, photos and memos',
            filled: true,
            fillColor: theme.colorScheme.surfaceContainerHighest,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(24),
              borderSide: BorderSide.none,
            ),
            contentPadding: const EdgeInsets.fromLTRB(20, 14, 8, 14),
            suffixIcon: Padding(
              padding: const EdgeInsets.only(right: 6),
              child: vm.asking
                  ? IconButton.filledTonal(
                      key: const Key('knowledge-ask-stop'),
                      tooltip: 'Stop',
                      onPressed: vm.stopAnswer,
                      icon: const Icon(Icons.stop_rounded),
                    )
                  : IconButton.filled(
                      key: const Key('knowledge-ask-send'),
                      tooltip: 'Ask',
                      onPressed: llm == null
                          ? null
                          : () => onAsk(controller.text),
                      icon: const Icon(Icons.arrow_upward_rounded),
                    ),
            ),
          ),
        ),
        const SizedBox(height: 12),
        _Suggestions(
          keyPrefix: 'knowledge-question',
          suggestions: kKnowledgeQuestions,
          onTap: llm == null ? (_) {} : onAsk,
        ),
        if (vm.question case final question?) ...[
          const SizedBox(height: 16),
          Align(
            alignment: Alignment.centerRight,
            child: Container(
              constraints: const BoxConstraints(maxWidth: 320),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(18),
                  topRight: Radius.circular(18),
                  bottomLeft: Radius.circular(18),
                  bottomRight: Radius.circular(4),
                ),
              ),
              child: Text(
                question,
                style: TextStyle(color: theme.colorScheme.onPrimaryContainer),
              ),
            ),
          ),
          if (vm.answerSources.isNotEmpty) ...[
            const _SectionHeader('FROM YOUR LIBRARY'),
            _SourcesStrip(vm: vm),
          ],
          const SizedBox(height: 12),
          _AnswerCard(vm: vm),
        ],
      ],
    );
  }
}

/// The retrieved items, numbered as the answer cites them.
class _SourcesStrip extends StatelessWidget {
  const _SourcesStrip({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    return ValueListenableBuilder<String>(
      valueListenable: vm.answer,
      builder: (context, answer, _) {
        final cited = citedSources(answer);
        return SizedBox(
          height: 64,
          child: ListView.separated(
            scrollDirection: Axis.horizontal,
            itemCount: vm.answerSources.length,
            separatorBuilder: (_, _) => const SizedBox(width: 8),
            itemBuilder: (context, i) => _SourceChip(
              vm: vm,
              source: vm.answerSources[i],
              cited: cited.contains(vm.answerSources[i].number),
            ),
          ),
        );
      },
    );
  }
}

class _SourceChip extends StatelessWidget {
  const _SourceChip({
    required this.vm,
    required this.source,
    required this.cited,
  });

  final KnowledgeViewModel vm;
  final KnowledgeSource source;
  final bool cited;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final item = source.item;
    final color = kindColor(item.kind, theme.brightness);
    return AnimatedContainer(
      key: Key('knowledge-source-${source.number}'),
      duration: const Duration(milliseconds: 300),
      width: 176,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: cited
            ? color.withValues(alpha: 0.18)
            : theme.colorScheme.surfaceContainerHigh,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: cited ? color : theme.colorScheme.outlineVariant,
          width: cited ? 1.5 : 1,
        ),
      ),
      child: Row(
        children: [
          switch (item) {
            PhotoItem(:final reference) => KnowledgeThumb(
              image: vm.imageFor(reference),
              size: 44,
            ),
            MemoItem() => MemoPlayButton(vm: vm, memo: item, size: 44),
            NoteItem() => const _KindBadge(kind: KnowledgeKind.note, size: 44),
          },
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  '[${source.number}] ${item.kind.label}'
                  '${source.attached ? ' · seen' : ''}',
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: color,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                Text(
                  item.title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.bodySmall,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// The streamed answer, its citations drawn as small numbered pills.
class _AnswerCard extends StatelessWidget {
  const _AnswerCard({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final llm = vm.setup?.llm;
    return Card(
      key: const Key('knowledge-answer'),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.auto_awesome,
                  size: 18,
                  color: theme.colorScheme.primary,
                ),
                const SizedBox(width: 6),
                Text(
                  llm?.name ?? 'Answer',
                  style: theme.textTheme.labelLarge?.copyWith(
                    color: theme.colorScheme.primary,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            ValueListenableBuilder<String>(
              valueListenable: vm.answer,
              builder: (context, answer, _) {
                if (answer.isEmpty && vm.asking) {
                  return _Thinking(vm: vm);
                }
                return _CitedText(
                  vm: vm,
                  text: answer.trim(),
                  streaming: vm.asking,
                );
              },
            ),
            if (_stats(vm) case final stats?) ...[
              const SizedBox(height: 12),
              Text(
                stats,
                key: const Key('knowledge-answer-stats'),
                style: theme.textTheme.labelSmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  static String? _stats(KnowledgeViewModel vm) {
    final parts = [
      if (vm.retrievalTook case final t?) 'retrieved in ${shortDuration(t)}',
      if (vm.firstWordsAfter case final t?) 'first words ${shortDuration(t)}',
      if (vm.answerTook case final t?) 'done in ${shortDuration(t)}',
      if (vm.llmBackend case final backend? when vm.llmState == LlmState.ready)
        'on $backend',
      if (vm.answerSources.any((s) => s.attached)) '1 photo seen',
    ];
    return parts.isEmpty ? null : parts.join(' · ');
  }
}

class _Thinking extends StatelessWidget {
  const _Thinking({required this.vm});

  final KnowledgeViewModel vm;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final text = switch (vm.llmState) {
      LlmState.loading => 'Loading ${vm.setup?.llm?.name ?? 'the model'}…',
      _ when vm.answerSources.any((s) => s.attached) =>
        'Reading the sources and looking at the photo…',
      _ => 'Reading the sources…',
    };
    return Row(
      children: [
        const SizedBox(
          width: 14,
          height: 14,
          child: CircularProgressIndicator(strokeWidth: 2),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            text,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
      ],
    );
  }
}

class _CitedText extends StatelessWidget {
  const _CitedText({
    required this.vm,
    required this.text,
    required this.streaming,
  });

  final KnowledgeViewModel vm;
  final String text;
  final bool streaming;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final spans = <InlineSpan>[];
    var last = 0;
    for (final match in RegExp(r'\[(\d+(?:\s*,\s*\d+)*)\]').allMatches(text)) {
      if (match.start > last) {
        spans.add(TextSpan(text: text.substring(last, match.start)));
      }
      for (final number in match.group(1)!.split(',')) {
        final n = int.tryParse(number.trim());
        final source = vm.answerSources.where((s) => s.number == n).firstOrNull;
        final color = source == null
            ? theme.colorScheme.outline
            : kindColor(source.item.kind, theme.brightness);
        spans.add(
          WidgetSpan(
            alignment: PlaceholderAlignment.middle,
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 2),
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                '${n ?? '?'}',
                style: theme.textTheme.labelSmall?.copyWith(
                  color: color,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ),
        );
      }
      last = match.end;
    }
    if (last < text.length) spans.add(TextSpan(text: text.substring(last)));
    if (streaming) {
      spans.add(
        WidgetSpan(
          alignment: PlaceholderAlignment.middle,
          child: _Cursor(color: theme.colorScheme.primary),
        ),
      );
    }
    return Text.rich(
      TextSpan(
        style: theme.textTheme.bodyLarge?.copyWith(height: 1.45),
        children: spans,
      ),
      key: const Key('knowledge-answer-text'),
    );
  }
}

class _Cursor extends StatefulWidget {
  const _Cursor({required this.color});

  final Color color;

  @override
  State<_Cursor> createState() => _CursorState();
}

class _CursorState extends State<_Cursor> with SingleTickerProviderStateMixin {
  late final AnimationController _controller = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 700),
  )..repeat(reverse: true);

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => FadeTransition(
    opacity: _controller,
    child: Container(
      width: 8,
      height: 16,
      margin: const EdgeInsets.only(left: 2),
      decoration: BoxDecoration(
        color: widget.color,
        borderRadius: BorderRadius.circular(2),
      ),
    ),
  );
}

// ── About ───────────────────────────────────────────────────────────────────

Future<void> showKnowledgeAbout(BuildContext context, KnowledgeViewModel vm) {
  final embedder = vm.setup?.embedder;
  return showModalBottomSheet<void>(
    context: context,
    showDragHandle: true,
    isScrollControlled: true,
    builder: (context) {
      final theme = Theme.of(context);
      Widget para(String text) => Padding(
        padding: const EdgeInsets.only(bottom: 10),
        child: Text(text, style: theme.textTheme.bodyMedium),
      );
      return SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('How it works', style: theme.textTheme.titleLarge),
              const SizedBox(height: 12),
              para(
                'Everything runs on this device. EmbeddingGemma 2'
                '${embedder == null ? '' : ' (${embedder.name})'} turns notes '
                '(their text), photos (their pixels) and voice memos (their '
                'sound) into vectors in ONE space, each kind in its own '
                'SQLite index (flutter_edge_ai_rag, sqlite-vec).',
              ),
              para(
                'A text query is compared with all three. Each kind has its '
                'own baseline similarity, so a hit is ranked by how far it '
                'stands out from the rest of its kind (its similarity minus '
                'the median of its kind); the bar shows that, the number is '
                'the raw cosine.',
              ),
              para(
                'Voice memos are found by their sound, not their transcript: '
                'the transcript (from the speech-to-text model) is only what '
                'you read, and what the language model reads.',
              ),
              para(
                'Ask retrieves the closest $kAnswerSources items, sends them '
                'to the language model as numbered sources — the best photo '
                'as an image, when the model sees images — and the answer '
                'cites them.',
              ),
            ],
          ),
        ),
      );
    },
  );
}
