import 'dart:async';

import 'package:flutter/material.dart';

import 'embedding_models_screen.dart';
import 'knowledge/knowledge_app_dependencies.dart';
import 'knowledge/knowledge_dependencies.dart';
import 'knowledge/knowledge_theme.dart';
import 'knowledge/knowledge_view_model.dart';
import 'knowledge/knowledge_widgets.dart';
import 'model_selection_screen.dart';

/// The Knowledge screen's tabs.
enum KnowledgeTab { library, search, live, ask }

/// Knowledge: a personal knowledge base — notes, photos and voice memos —
/// searched by text, by photo and by what is said in a meeting, and asked
/// questions, all on the device.
///
/// EmbeddingGemma 2 (the ACTIVE embedder; the 740M embeds text, images and
/// audio into one space) indexes each kind into its own RAG index on SQLite;
/// nothing is transcribed or captioned. The active language model (Gemma 4)
/// answers from what the search retrieves, with the retrieved photo and
/// recording sent along, and in Live hears the meeting window by window.
/// Each model is chosen on its model list; the screen links there when one is
/// missing.
class KnowledgeScreen extends StatefulWidget {
  const KnowledgeScreen({super.key, this.dependencies, this.openModelList});

  /// Defaults to [knowledgeAppDependencies]; tests pass fakes.
  final KnowledgeDependencies? dependencies;

  /// Opens a model list. Defaults to pushing the app's list screen.
  final Future<void> Function(BuildContext context, KnowledgeModelList list)?
  openModelList;

  @override
  State<KnowledgeScreen> createState() => _KnowledgeScreenState();
}

class _KnowledgeScreenState extends State<KnowledgeScreen> {
  late final KnowledgeViewModel _vm = KnowledgeViewModel(
    widget.dependencies ?? knowledgeAppDependencies(),
  );
  final _query = TextEditingController();
  final _question = TextEditingController();
  KnowledgeTab _tab = KnowledgeTab.library;
  bool _tabChosen = false;

  @override
  void initState() {
    super.initState();
    // open() keeps every failure in the view model's state, for the screen.
    unawaited(_vm.open());
  }

  @override
  void dispose() {
    _vm.dispose();
    _query.dispose();
    _question.dispose();
    super.dispose();
  }

  Future<void> _openList(KnowledgeModelList list) async {
    final open = widget.openModelList;
    if (open != null) {
      await open(context, list);
    } else {
      final screen = switch (list) {
        KnowledgeModelList.embedder => const EmbeddingModelsScreen(),
        KnowledgeModelList.llm => const ModelSelectionScreen(),
      };
      await Navigator.push<void>(
        context,
        MaterialPageRoute<void>(builder: (_) => screen),
      );
    }
    if (!mounted) return;
    await _vm.open();
  }

  void _search(String text) {
    _query.text = text;
    FocusScope.of(context).unfocus();
    unawaited(_vm.search(text));
  }

  void _ask(String text) {
    if (text.trim().isEmpty) return;
    // The question moves into the conversation; the field is for the next.
    _question.clear();
    FocusScope.of(context).unfocus();
    unawaited(_vm.ask(text));
  }

  @override
  Widget build(BuildContext context) {
    final brightness = Theme.of(context).brightness;
    return Theme(
      data: knowledgeTheme(brightness),
      child: ListenableBuilder(
        listenable: _vm,
        builder: (context, _) {
          final tabs = [
            KnowledgeTab.library,
            KnowledgeTab.search,
            if (_vm.setup?.liveSupported ?? false) KnowledgeTab.live,
            KnowledgeTab.ask,
          ];
          // The first time it is ready, an index with items still to add
          // opens on Library (to index them), a full one on Search.
          if (!_tabChosen && _vm.status == KnowledgeStatus.ready) {
            _tab = _vm.pending.isEmpty
                ? KnowledgeTab.search
                : KnowledgeTab.library;
            _tabChosen = true;
          }
          if (!tabs.contains(_tab)) _tab = KnowledgeTab.library;
          return Scaffold(
            appBar: AppBar(
              title: const KnowledgeTitle(),
              actions: [
                IconButton(
                  key: const Key('knowledge-about'),
                  tooltip: 'How it works',
                  icon: const Icon(Icons.info_outline),
                  onPressed: () => showKnowledgeAbout(context, _vm),
                ),
              ],
            ),
            body: SafeArea(
              top: false,
              child: switch (_vm.status) {
                KnowledgeStatus.loading => const KnowledgeLoading(),
                KnowledgeStatus.failed => KnowledgeSetupProblem(
                  vm: _vm,
                  onOpenList: _openList,
                ),
                KnowledgeStatus.ready => IndexedStack(
                  index: tabs.indexOf(_tab),
                  children: [
                    for (final tab in tabs)
                      switch (tab) {
                        KnowledgeTab.library => LibraryTab(
                          vm: _vm,
                          onOpenList: _openList,
                        ),
                        KnowledgeTab.search => SearchTab(
                          vm: _vm,
                          controller: _query,
                          onSearch: _search,
                        ),
                        KnowledgeTab.live => LiveTab(vm: _vm),
                        KnowledgeTab.ask => AskTab(
                          vm: _vm,
                          controller: _question,
                          onAsk: _ask,
                          onOpenList: _openList,
                        ),
                      },
                  ],
                ),
              },
            ),
            bottomNavigationBar: _vm.status != KnowledgeStatus.ready
                ? null
                : NavigationBar(
                    selectedIndex: tabs.indexOf(_tab),
                    onDestinationSelected: (i) => setState(() {
                      _tab = tabs[i];
                      _tabChosen = true;
                    }),
                    destinations: [
                      for (final tab in tabs)
                        switch (tab) {
                          KnowledgeTab.library => NavigationDestination(
                            key: const Key('knowledge-tab-library'),
                            icon: _PendingBadge(
                              pending: _vm.pending.length,
                              child: const Icon(
                                Icons.collections_bookmark_outlined,
                              ),
                            ),
                            selectedIcon: _PendingBadge(
                              pending: _vm.pending.length,
                              child: const Icon(Icons.collections_bookmark),
                            ),
                            label: 'Library',
                          ),
                          KnowledgeTab.search => const NavigationDestination(
                            key: Key('knowledge-tab-search'),
                            icon: Icon(Icons.search),
                            label: 'Search',
                          ),
                          KnowledgeTab.live => const NavigationDestination(
                            key: Key('knowledge-tab-live'),
                            icon: Icon(Icons.hearing_outlined),
                            selectedIcon: Icon(Icons.hearing),
                            label: 'Live',
                          ),
                          KnowledgeTab.ask => const NavigationDestination(
                            key: Key('knowledge-tab-ask'),
                            icon: Icon(Icons.auto_awesome_outlined),
                            selectedIcon: Icon(Icons.auto_awesome),
                            label: 'Ask',
                          ),
                        },
                    ],
                  ),
          );
        },
      ),
    );
  }
}

/// How many items are not indexed yet, on the Library tab.
class _PendingBadge extends StatelessWidget {
  const _PendingBadge({required this.pending, required this.child});

  final int pending;
  final Widget child;

  @override
  Widget build(BuildContext context) => Badge(
    key: pending == 0 ? null : const Key('knowledge-pending'),
    isLabelVisible: pending > 0,
    label: Text('$pending'),
    child: child,
  );
}
