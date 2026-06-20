"use client";

import { useState, useEffect, useCallback, useRef, type InputHTMLAttributes } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useDropzone } from "react-dropzone";
import confetti from "canvas-confetti";
import {
  listCases,
  createCase,
  deleteCase,
  getCaseDocuments,
  addDocumentToCase,
  removeDocumentFromCase,
  listDocuments,
  uploadDocument,
  deleteDocument,
  bulkUploadCase,
  getCaseJobs,
  Case,
  CaseDocument,
  Document,
  CaseJobStatus,
} from "@/lib/api";

function truncate(str: string, n = 50) {
  const stem = str.replace(/\.(pdf|pptx)$/i, "");
  return stem.length > n ? stem.slice(0, n) + "…" : stem;
}

function isSupportedDocument(file: File) {
  return /\.(pdf|pptx)$/i.test(file.name);
}

function Spinner() {
  return (
    <div className="flex gap-1.5">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}

const folderPickerAttrs = {
  webkitdirectory: "",
  directory: "",
} as InputHTMLAttributes<HTMLInputElement>;

function notifyCasesChanged() {
  window.dispatchEvent(new CustomEvent("atticus-cases-changed"));
}

export default function CasesPage() {
  const { user, idToken, loading } = useAuth();

  const [tab, setTab] = useState<"cases" | "documents">("cases");

  const [cases, setCases] = useState<Case[]>([]);
  const [casesLoading, setCasesLoading] = useState(false);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [caseDocuments, setCaseDocuments] = useState<CaseDocument[]>([]);
  const [caseDocsLoading, setCaseDocsLoading] = useState(false);
  const [newCaseName, setNewCaseName] = useState("");
  const [newCaseDesc, setNewCaseDesc] = useState("");
  const [showCreateCase, setShowCreateCase] = useState(false);
  const [caseError, setCaseError] = useState<string | null>(null);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [docError, setDocError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [selectedDocIds, setSelectedDocIds] = useState<Set<string>>(new Set());

  // Bulk upload modal state
  const [showBulkModal, setShowBulkModal] = useState(false);
  const [bulkCaseName, setBulkCaseName] = useState("");
  const [bulkMatterType, setBulkMatterType] = useState("");
  const [bulkJurisdiction, setBulkJurisdiction] = useState("");
  const [bulkClientName, setBulkClientName] = useState("");
  const [bulkNotes, setBulkNotes] = useState("");
  const [bulkFiles, setBulkFiles] = useState<File[]>([]);
  const [bulkUploading, setBulkUploading] = useState(false);
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [bulkCaseId, setBulkCaseId] = useState<string | null>(null);
  const [bulkJobStatus, setBulkJobStatus] = useState<CaseJobStatus | null>(null);
  const [bulkStartedAt, setBulkStartedAt] = useState<number | null>(null);
  const bulkFileRef = useRef<HTMLInputElement>(null);

  const loadCases = useCallback(async () => {
    setCasesLoading(true);
    try {
      const cs = await listCases();
      setCases(cs);
    } catch {
      setCaseError("Failed to load cases.");
    } finally {
      setCasesLoading(false);
    }
  }, []);

  const loadDocuments = useCallback(async () => {
    setDocsLoading(true);
    try {
      const docs = await listDocuments();
      setDocuments(docs);
    } catch {
      setDocError("Failed to load documents.");
    } finally {
      setDocsLoading(false);
    }
  }, []);

  const loadCaseDocuments = useCallback(
    async (caseId: string) => {
      setCaseDocsLoading(true);
      try {
        const docs = await getCaseDocuments(caseId);
        setCaseDocuments(docs);
      } catch {
        setCaseDocuments([]);
      } finally {
        setCaseDocsLoading(false);
      }
    },
    []
  );

  useEffect(() => {
    if (user) {
      loadCases();
      loadDocuments();
    }
  }, [user, loadCases, loadDocuments]);

  useEffect(() => {
    if (selectedCaseId) loadCaseDocuments(selectedCaseId);
    setSelectedDocIds(new Set());
  }, [selectedCaseId, loadCaseDocuments]);

  // Poll job status while jobs are pending/running
  useEffect(() => {
    if (!bulkCaseId || !bulkJobStatus) return;
    if (bulkJobStatus.pending + bulkJobStatus.in_progress === 0) return;
    const interval = setInterval(() => {
      getCaseJobs(bulkCaseId).then(setBulkJobStatus).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, [bulkCaseId, bulkJobStatus]);

  // Reload cases list when bulk upload finishes
  useEffect(() => {
    if (!bulkJobStatus) return;
    if (bulkJobStatus.pending + bulkJobStatus.in_progress === 0 && bulkJobStatus.total > 0) {
      loadCases();
      notifyCasesChanged();
    }
  }, [bulkJobStatus, loadCases]);

  function resetBulkModal() {
    setBulkCaseName("");
    setBulkMatterType("");
    setBulkJurisdiction("");
    setBulkClientName("");
    setBulkNotes("");
    setBulkFiles([]);
    setBulkError(null);
    setBulkCaseId(null);
    setBulkJobStatus(null);
    setBulkStartedAt(null);
    setBulkUploading(false);
    if (bulkFileRef.current) bulkFileRef.current.value = "";
  }

  function handleBulkFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(e.target.files ?? []).filter(isSupportedDocument);
    const firstFolder = (selected[0] as (File & { webkitRelativePath?: string }) | undefined)
      ?.webkitRelativePath
      ?.split("/")
      .filter(Boolean)[0];
    if (!bulkCaseName.trim() && firstFolder) {
      setBulkCaseName(firstFolder);
    }
    setBulkFiles(selected);
  }

  async function handleBulkUpload() {
    if (!bulkCaseName.trim() || bulkFiles.length === 0) return;
    setBulkUploading(true);
    setBulkError(null);
    setBulkStartedAt(Date.now());
    try {
      const result = await bulkUploadCase(bulkFiles, {
        case_name: bulkCaseName.trim(),
        matter_type: bulkMatterType || undefined,
        jurisdiction: bulkJurisdiction || undefined,
        client_name: bulkClientName || undefined,
        notes: bulkNotes || undefined,
      });
      setBulkCaseId(result.case_id);
      const initial: CaseJobStatus = {
        total: result.total_files,
        completed: 0,
        failed: 0,
        in_progress: 0,
        pending: result.total_files,
        errors: result.skipped.map((f) => `${f}: skipped (too large or unsupported)`),
      };
      setBulkJobStatus(initial);
      await loadCases();
      notifyCasesChanged();
    } catch (err) {
      setBulkError(err instanceof Error ? err.message : "Bulk upload failed.");
      setBulkUploading(false);
    }
  }

  async function handleCreateCase(e: React.FormEvent) {
    e.preventDefault();
    if (!idToken || !newCaseName.trim()) return;
    setCaseError(null);
    try {
      await createCase(idToken, {
        case_name: newCaseName.trim(),
        description: newCaseDesc.trim() || undefined,
      });
      setNewCaseName("");
      setNewCaseDesc("");
      setShowCreateCase(false);
      await loadCases();
      notifyCasesChanged();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to create case.";
      if (msg === "TRIAL_CASE_LIMIT") {
        setCaseError("Trial accounts are limited to 2 cases. Upgrade to add unlimited cases.");
      } else {
        setCaseError("Failed to create case.");
      }
    }
  }

  async function handleDeleteCase(caseId: string) {
    if (!idToken) return;
    if (!confirm("Delete this case? Documents are not deleted.")) return;
    try {
      await deleteCase(idToken, caseId);
      if (selectedCaseId === caseId) setSelectedCaseId(null);
      await loadCases();
      notifyCasesChanged();
    } catch {
      setCaseError("Failed to delete case.");
    }
  }

  async function handleRemoveDocFromCase(docId: string) {
    if (!idToken || !selectedCaseId) return;
    try {
      await removeDocumentFromCase(idToken, selectedCaseId, docId);
      await loadCaseDocuments(selectedCaseId);
      await loadCases();
    } catch {
      setCaseError("Failed to remove document.");
    }
  }

  async function handleAddDocuments() {
    if (!idToken || !selectedCaseId || selectedDocIds.size === 0) return;
    setCaseError(null);
    try {
      for (const docId of selectedDocIds) {
        await addDocumentToCase(idToken, selectedCaseId, docId);
      }
      setSelectedDocIds(new Set());
      await loadCaseDocuments(selectedCaseId);
      await loadCases();
    } catch {
      setCaseError("Failed to add documents.");
    }
  }

  async function doUpload(file: File) {
    if (!idToken) return;
    if (!isSupportedDocument(file)) {
      setUploadError("Only PDF and PPTX files are supported.");
      return;
    }
    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);
    try {
      const result = await uploadDocument(idToken, file);
      setUploadSuccess(
        `Uploaded "${result.file_name}" — ${result.page_count} pages, ${result.chunks_inserted} chunks.`
      );
      await loadDocuments();

      // Confetti on first upload
      if (typeof window !== "undefined" && !localStorage.getItem("first_upload_done")) {
        localStorage.setItem("first_upload_done", "true");
        confetti({
          particleCount: 100,
          spread: 70,
          origin: { y: 0.6 },
          colors: ["#6366f1", "#a855f7", "#C9A84C"],
        });
      }
    } catch (err: unknown) {
      setUploadError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    await doUpload(file);
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
    },
    disabled: uploading,
    noClick: true,
    onDrop: (files) => {
      const supported = files.find(isSupportedDocument);
      if (supported) doUpload(supported);
      else setUploadError("Only PDF and PPTX files are supported.");
    },
  });

  async function handleDeleteDocument(docId: string) {
    if (!idToken) return;
    if (!confirm("Delete this document? This also removes it from all cases.")) return;
    try {
      await deleteDocument(idToken, docId);
      await loadDocuments();
    } catch {
      setDocError("Failed to delete document.");
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner />
      </div>
    );
  }

  if (!user) return null;

  const selectedCase = cases.find((c) => c.case_id === selectedCaseId);
  const caseDocIds = new Set(caseDocuments.map((d) => d.document_id));
  const userDocs = documents.filter((d) => !d.is_system);
  const systemDocs = documents.filter((d) => d.is_system);
  const availableDocs = userDocs.filter((d) => !caseDocIds.has(d.document_id));

  return (
    <div className="flex flex-col h-full">
      {/* Tab header */}
      <div className="flex items-center gap-0 border-b border-white/[0.07] px-6 pt-5 shrink-0">
        {(["cases", "documents"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-5 py-2.5 text-sm font-medium transition-colors relative capitalize ${
              tab === t
                ? "text-indigo-300"
                : "text-zinc-500 hover:text-zinc-300"
            }`}
          >
            {t === "cases" ? "My Cases" : "Documents"}
            {tab === t && (
              <div className="absolute bottom-0 left-0 right-0 h-px bg-indigo-500" />
            )}
          </button>
        ))}
        <div className="ml-auto pb-2">
          <button
            onClick={() => { resetBulkModal(); setShowBulkModal(true); }}
            className="flex items-center gap-1.5 text-xs text-indigo-300 bg-indigo-500/10 border border-indigo-500/25 hover:bg-indigo-500/20 rounded-lg px-3 py-1.5 transition-colors"
          >
            📁 Bulk Upload
          </button>
        </div>
      </div>

      {/* Cases tab */}
      {tab === "cases" && (
        <div className="flex flex-1 overflow-hidden">
          {/* Case list */}
          <div className="w-64 border-r border-white/[0.07] flex flex-col shrink-0">
            <div className="px-4 py-4 border-b border-white/[0.06]">
              <button
                onClick={() => setShowCreateCase((v) => !v)}
                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg px-4 py-2.5 transition-colors"
              >
                + New Case
              </button>
            </div>

            {showCreateCase && (
              <form onSubmit={handleCreateCase} className="px-4 py-4 border-b border-white/[0.06] space-y-2">
                <input
                  autoFocus
                  value={newCaseName}
                  onChange={(e) => setNewCaseName(e.target.value)}
                  placeholder="Case name"
                  className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
                />
                <textarea
                  value={newCaseDesc}
                  onChange={(e) => setNewCaseDesc(e.target.value)}
                  placeholder="Description (optional)"
                  rows={2}
                  className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 resize-none"
                />
                <div className="flex gap-2">
                  <button
                    type="submit"
                    disabled={!newCaseName.trim()}
                    className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg py-1.5 transition-colors"
                  >
                    Create
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowCreateCase(false)}
                    className="text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-3 py-1.5"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            )}

            <div className="flex-1 overflow-y-auto py-3 px-3 space-y-0.5">
              {casesLoading && cases.length === 0 && (
                <div className="flex justify-center py-6">
                  <Spinner />
                </div>
              )}

              {/* Empty state for cases */}
              {!casesLoading && cases.length === 0 && (
                <div className="flex flex-col items-center justify-center py-12 px-4">
                  <div className="text-5xl mb-4">📂</div>
                  <p className="text-sm text-zinc-400 mb-2 font-medium">No cases yet</p>
                  <p className="text-xs text-zinc-600 text-center mb-4">
                    Create your first case to organize your documents and research.
                  </p>
                  <button
                    onClick={() => setShowCreateCase(true)}
                    className="bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg px-4 py-2 transition-colors"
                  >
                    Create your first case
                  </button>
                </div>
              )}

              {cases.map((c) => (
                <div
                  key={c.case_id}
                  onClick={() => setSelectedCaseId(c.case_id)}
                  className={`group flex items-start gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${
                    selectedCaseId === c.case_id
                      ? "bg-indigo-500/15 text-indigo-300"
                      : "hover:bg-white/[0.04] text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium truncate">{c.case_name}</div>
                    <div className="text-[0.65rem] text-zinc-600 mt-0.5">
                      {c.document_count} doc{c.document_count !== 1 ? "s" : ""}
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteCase(c.case_id); }}
                    className="opacity-0 group-hover:opacity-100 text-zinc-700 hover:text-red-400 text-xs mt-0.5 transition-opacity shrink-0"
                    title="Delete case"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Case detail */}
          <div className="flex-1 overflow-y-auto p-6">
            {!selectedCaseId && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="text-3xl mb-3">📁</div>
                  <p className="text-sm text-zinc-600">Select a case to view details</p>
                </div>
              </div>
            )}

            {selectedCaseId && selectedCase && (
              <div className="max-w-3xl">
                <div className="mb-5">
                  <h2 className="text-lg font-bold text-white">{selectedCase.case_name}</h2>
                  {selectedCase.description && (
                    <p className="text-sm text-zinc-500 mt-1">{selectedCase.description}</p>
                  )}
                </div>

                {caseError && (
                  <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300 mb-4">
                    {caseError}
                    <button onClick={() => setCaseError(null)} className="ml-2 text-red-400">×</button>
                  </div>
                )}

                <div className="mb-6">
                  <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">
                    DOCUMENTS IN THIS CASE
                  </p>
                  {caseDocsLoading && <div className="flex py-4"><Spinner /></div>}
                  {!caseDocsLoading && caseDocuments.length === 0 && (
                    <p className="text-xs text-zinc-600">No documents yet. Add some below.</p>
                  )}
                  {caseDocuments.map((doc) => (
                    <div
                      key={doc.document_id}
                      className="flex items-center justify-between bg-white/[0.03] border border-white/[0.07] rounded-lg px-4 py-2.5 mb-1.5"
                    >
                      <div className="min-w-0">
                        <span className="text-xs text-zinc-300 font-medium truncate block">
                          📄 {truncate(doc.file_name)}
                        </span>
                        {doc.company && (
                          <span className="text-[0.65rem] text-zinc-600">{doc.company}</span>
                        )}
                      </div>
                      <button
                        onClick={() => handleRemoveDocFromCase(doc.document_id)}
                        className="text-xs text-zinc-600 hover:text-red-400 transition-colors ml-4 shrink-0"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>

                {availableDocs.length > 0 && (
                  <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 mb-6">
                    <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">
                      ADD DOCUMENTS
                    </p>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto mb-3">
                      {availableDocs.map((doc) => (
                        <label
                          key={doc.document_id}
                          className="flex items-center gap-2.5 px-3 py-2 rounded-lg hover:bg-white/[0.04] cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={selectedDocIds.has(doc.document_id)}
                            onChange={(e) => {
                              const next = new Set(selectedDocIds);
                              if (e.target.checked) next.add(doc.document_id);
                              else next.delete(doc.document_id);
                              setSelectedDocIds(next);
                            }}
                            className="accent-indigo-500"
                          />
                          <span className="text-xs text-zinc-400 truncate">
                            📄 {truncate(doc.file_name)}
                            {doc.company && (
                              <span className="text-zinc-600 ml-1">· {doc.company}</span>
                            )}
                            {doc.is_system && (
                              <span className="text-zinc-700 ml-1">[system]</span>
                            )}
                          </span>
                        </label>
                      ))}
                    </div>
                    <button
                      onClick={handleAddDocuments}
                      disabled={selectedDocIds.size === 0}
                      className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-medium rounded-lg px-4 py-2 transition-colors"
                    >
                      Add selected ({selectedDocIds.size})
                    </button>
                  </div>
                )}

                <div className="border border-red-500/20 rounded-xl p-4">
                  <p className="text-xs font-semibold text-red-400 mb-1">Danger zone</p>
                  <p className="text-xs text-zinc-600 mb-3">
                    Deletes the case and all document associations. Documents themselves are not deleted.
                  </p>
                  <button
                    onClick={() => handleDeleteCase(selectedCaseId)}
                    className="text-xs text-red-400 border border-red-500/30 hover:bg-red-500/10 rounded-lg px-4 py-2 transition-colors"
                  >
                    Delete case
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Documents tab */}
      {/* Bulk Upload Modal */}
      {showBulkModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-zinc-900 border border-white/[0.1] rounded-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/[0.08]">
              <h2 className="text-sm font-semibold text-white">📁 Bulk Upload</h2>
              <button
                onClick={() => { setShowBulkModal(false); resetBulkModal(); }}
                className="text-zinc-600 hover:text-zinc-300 text-lg leading-none"
              >×</button>
            </div>

            <div className="px-6 py-5 space-y-4">
              {!bulkCaseId ? (
                <>
                  {/* Case metadata */}
                  <div>
                    <label className="block text-xs text-zinc-500 mb-1">Case Name *</label>
                    <input
                      value={bulkCaseName}
                      onChange={(e) => setBulkCaseName(e.target.value)}
                      placeholder="e.g. Smith v. Jones"
                      className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-zinc-500 mb-1">Matter Type</label>
                      <select
                        value={bulkMatterType}
                        onChange={(e) => setBulkMatterType(e.target.value)}
                        className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-zinc-300 focus:outline-none focus:border-indigo-500/50"
                      >
                        <option value="">Select…</option>
                        <option>Civil Litigation</option>
                        <option>Criminal Defense</option>
                        <option>Contract Review</option>
                        <option>Regulatory</option>
                        <option>Other</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-500 mb-1">Jurisdiction</label>
                      <input
                        value={bulkJurisdiction}
                        onChange={(e) => setBulkJurisdiction(e.target.value)}
                        placeholder="e.g. S.D.N.Y."
                        className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs text-zinc-500 mb-1">Client Name</label>
                    <input
                      value={bulkClientName}
                      onChange={(e) => setBulkClientName(e.target.value)}
                      placeholder="Optional"
                      className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-zinc-500 mb-1">Notes</label>
                    <textarea
                      value={bulkNotes}
                      onChange={(e) => setBulkNotes(e.target.value)}
                      rows={2}
                      placeholder="Optional notes"
                      className="w-full bg-white/[0.05] border border-white/[0.1] rounded-lg px-3 py-2 text-sm text-white placeholder-zinc-600 focus:outline-none focus:border-indigo-500/50 resize-none"
                    />
                  </div>

                  {/* File picker */}
                  <div>
                    <label className="block text-xs text-zinc-500 mb-1">PDF / PPTX Files</label>
                    <label className="flex flex-col items-center gap-2 border-2 border-dashed border-white/[0.12] hover:border-indigo-500/40 rounded-xl px-5 py-5 cursor-pointer transition-colors">
                      <span className="text-2xl">📂</span>
                      <span className="text-sm text-zinc-400">
                        {bulkFiles.length > 0
                          ? `${bulkFiles.length} file${bulkFiles.length !== 1 ? "s" : ""} selected (${(bulkFiles.reduce((s, f) => s + f.size, 0) / 1024 / 1024).toFixed(1)} MB)`
                          : "Click to select a folder or multiple PDFs/PPTXs"}
                      </span>
                      <span className="text-xs text-zinc-600">PDF/PPTX files · max 50 MB each</span>
                      <input
                        {...folderPickerAttrs}
                        ref={bulkFileRef}
                        type="file"
                        accept=".pdf,.pptx"
                        multiple
                        className="hidden"
                        onChange={handleBulkFileChange}
                      />
                    </label>
                  </div>

                  {bulkError && (
                    <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300">
                      {bulkError}
                    </div>
                  )}

                  <div className="flex gap-2 pt-1">
                    <button
                      onClick={handleBulkUpload}
                      disabled={bulkUploading || !bulkCaseName.trim() || bulkFiles.length === 0}
                      className="flex-1 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg py-2.5 transition-colors"
                    >
                      {bulkUploading ? "Uploading…" : `Upload ${bulkFiles.length > 0 ? `${bulkFiles.length} file${bulkFiles.length !== 1 ? "s" : ""}` : "files"}`}
                    </button>
                    <button
                      onClick={() => { setShowBulkModal(false); resetBulkModal(); }}
                      className="text-xs text-zinc-600 hover:text-zinc-400 border border-white/[0.07] rounded-lg px-4"
                    >
                      Cancel
                    </button>
                  </div>
                </>
              ) : (
                /* Progress view */
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs text-zinc-500 mb-1.5">
                      <span>Ingestion progress</span>
                      <span>
                        {bulkJobStatus ? `${bulkJobStatus.completed + bulkJobStatus.failed} / ${bulkJobStatus.total}` : "…"}
                      </span>
                    </div>
                    <div className="h-2 bg-white/[0.06] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                        style={{
                          width: bulkJobStatus && bulkJobStatus.total > 0
                            ? `${((bulkJobStatus.completed + bulkJobStatus.failed) / bulkJobStatus.total) * 100}%`
                            : "0%",
                        }}
                      />
                    </div>
                  </div>

                  {bulkJobStatus && (
                    <div className="grid grid-cols-3 gap-2 text-center">
                      {[
                        { label: "Completed", value: bulkJobStatus.completed, color: "text-emerald-300" },
                        { label: "In progress", value: bulkJobStatus.in_progress + bulkJobStatus.pending, color: "text-indigo-300" },
                        { label: "Failed", value: bulkJobStatus.failed, color: "text-red-300" },
                      ].map(({ label, value, color }) => (
                        <div key={label} className="bg-white/[0.03] border border-white/[0.07] rounded-lg p-3">
                          <div className={`text-xl font-bold ${color}`}>{value}</div>
                          <div className="text-[0.65rem] text-zinc-600 mt-0.5">{label}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {bulkJobStatus && bulkStartedAt && bulkJobStatus.completed > 0 && (
                    <p className="text-xs text-zinc-600">
                      {(() => {
                        const elapsed = (Date.now() - bulkStartedAt) / 1000;
                        const rate = bulkJobStatus.completed / elapsed;
                        const remaining = bulkJobStatus.pending + bulkJobStatus.in_progress;
                        const eta = rate > 0 ? Math.ceil(remaining / rate) : null;
                        return eta !== null ? `ETA: ~${eta}s remaining` : null;
                      })()}
                    </p>
                  )}

                  {bulkJobStatus && bulkJobStatus.errors.length > 0 && (
                    <div className="space-y-1">
                      <p className="text-xs font-semibold text-red-400">Errors ({bulkJobStatus.errors.length})</p>
                      <div className="max-h-28 overflow-y-auto space-y-0.5">
                        {bulkJobStatus.errors.map((e, i) => (
                          <p key={i} className="text-[0.65rem] text-red-400/70 leading-relaxed">{e}</p>
                        ))}
                      </div>
                    </div>
                  )}

                  {bulkJobStatus && bulkJobStatus.pending + bulkJobStatus.in_progress === 0 ? (
                    <button
                      onClick={() => { setShowBulkModal(false); resetBulkModal(); }}
                      className="w-full bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg py-2.5 transition-colors"
                    >
                      Done
                    </button>
                  ) : (
                    <p className="text-xs text-zinc-600 text-center animate-pulse">Processing files in the background…</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {tab === "documents" && (
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-3xl">
            {/* Upload with drag-and-drop */}
            <div
              {...getRootProps()}
              className="bg-white/[0.02] border border-white/[0.07] rounded-xl p-5 mb-6"
            >
              <p className="text-sm font-semibold text-white mb-1">Upload Document</p>
              <p className="text-xs text-zinc-600 mb-4">
                Upload a PDF or PPTX to make it searchable across all chats and cases.
                Ingestion may take 1–3 minutes depending on document size.
              </p>

              {uploadError && (
                <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300 mb-3">
                  {uploadError}
                </div>
              )}
              {uploadSuccess && (
                <div className="bg-emerald-500/10 border border-emerald-500/25 rounded-lg px-4 py-2.5 text-xs text-emerald-300 mb-3">
                  {uploadSuccess}
                </div>
              )}

              <label
                className={`flex items-center gap-3 border-2 border-dashed rounded-xl px-5 py-6 cursor-pointer transition-all ${
                  isDragActive
                    ? "border-indigo-400 bg-indigo-500/10 scale-[1.01]"
                    : uploading
                    ? "border-indigo-500/40 opacity-60"
                    : "border-white/[0.12] hover:border-indigo-500/40"
                }`}
                style={isDragActive ? { borderStyle: "dashed", animation: "dash-march 0.5s linear infinite" } : undefined}
              >
                <input
                  {...getInputProps()}
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.pptx"
                  disabled={uploading}
                  onChange={handleUpload}
                  className="hidden"
                />
                <div className="text-2xl">{isDragActive ? "📥" : "📄"}</div>
                <div>
                  <p className="text-sm text-zinc-300">
                    {isDragActive
                      ? "Drop your PDF/PPTX here"
                      : uploading
                      ? "Uploading and ingesting…"
                      : "Click or drag a PDF/PPTX file here"}
                  </p>
                  <p className="text-xs text-zinc-600 mt-0.5">PDF/PPTX files</p>
                </div>
                {uploading && (
                  <div className="ml-auto">
                    <Spinner />
                  </div>
                )}
              </label>
            </div>

            {docError && (
              <div className="bg-red-500/10 border border-red-500/25 rounded-lg px-4 py-2.5 text-xs text-red-300 mb-4">
                {docError}
              </div>
            )}

            {/* My Documents */}
            <div className="mb-6">
              <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">
                MY DOCUMENTS {userDocs.length > 0 && `(${userDocs.length})`}
              </p>
              {docsLoading && <div className="flex py-4"><Spinner /></div>}
              {!docsLoading && userDocs.length === 0 && (
                <p className="text-xs text-zinc-600">No documents uploaded yet.</p>
              )}
              {userDocs.map((doc) => (
                <div
                  key={doc.document_id}
                  className="flex items-center justify-between bg-white/[0.03] border border-white/[0.07] rounded-lg px-4 py-3 mb-1.5"
                >
                  <div className="min-w-0">
                    <p className="text-xs text-zinc-200 font-medium truncate">
                      📄 {truncate(doc.file_name, 60)}
                    </p>
                    <p className="text-[0.65rem] text-zinc-600 mt-0.5">
                      {[doc.company, doc.document_type, doc.page_count ? `${doc.page_count}pp` : null]
                        .filter(Boolean)
                        .join(" · ")}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDeleteDocument(doc.document_id)}
                    className="text-xs text-zinc-600 hover:text-red-400 transition-colors ml-4 shrink-0"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>

            {/* System Documents */}
            {systemDocs.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-zinc-500 tracking-widest mb-3">
                  SYSTEM DOCUMENTS ({systemDocs.length})
                </p>
                <p className="text-xs text-zinc-600 mb-3">Shared documents available to all users.</p>
                {systemDocs.map((doc) => (
                  <div
                    key={doc.document_id}
                    className="flex items-center bg-white/[0.02] border border-white/[0.05] rounded-lg px-4 py-2.5 mb-1"
                  >
                    <div className="min-w-0">
                      <p className="text-xs text-zinc-400 truncate">
                        📄 {truncate(doc.file_name, 60)}
                        {doc.company && <span className="text-zinc-600 ml-1">· {doc.company}</span>}
                        {doc.page_count > 0 && (
                          <span className="text-zinc-700 ml-1">· {doc.page_count}pp</span>
                        )}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
