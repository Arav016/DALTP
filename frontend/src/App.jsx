import { useEffect, useMemo, useRef, useState } from "react";

const AUTH_TOKEN_KEY = "daltp.authToken";

const initialDashboard = {
  stats: [
    { label: "Evaluation runs", value: 0, subtext: "saved DALTP output runs", tone: "green" },
    { label: "Best BERTScore F1", value: "0.0000", subtext: "across scored runs", tone: "green" },
    { label: "Benchmark size", value: 0, subtext: "held-out QA samples", tone: "green" },
    { label: "Prepared bundles", value: 0, subtext: "local + Colab launch packages", tone: "amber" },
  ],
  recentRuns: [],
  defaultRunId: null,
  adapterSizeMb: 84,
  datasetCount: 0,
};

const initialSummary = {
  runId: null,
  runName: null,
  benchmarkSize: 0,
  systems: [],
};

const initialOptions = {
  models: [
    {
      id: "meta-llama/Meta-Llama-3.1-8B-Instruct",
      name: "Llama 3.1 8B",
      provider: "Meta",
      params: "8B params",
      vramHint: "16GB VRAM with QLoRA",
    },
  ],
  peftMethods: [
    { id: "qlora", label: "QLoRA (4-bit)" },
    { id: "lora", label: "LoRA" },
  ],
  loraRanks: [8, 16, 32],
  executionModes: [
    { id: "local", label: "Local", description: "Run directly on the user's machine." },
    { id: "colab", label: "Colab-assisted", description: "Package artifacts for Google Colab execution." },
  ],
  configPresets: [
    {
      id: "balanced-qlora",
      label: "Balanced QLoRA",
      description: "Default DALTP preset tuned for practical Colab runs with good benchmark quality.",
      manualDefaults: {
        learningRate: 0.0001,
        epochs: 3,
        batchSize: 1,
        gradientAccumulationSteps: 8,
        maxLength: 2048,
        dtype: "bfloat16",
        loadIn4bit: true,
      },
    },
    {
      id: "fast-iteration",
      label: "Fast iteration",
      description: "Smaller run for quick UI and workflow iteration.",
      manualDefaults: {
        learningRate: 0.00015,
        epochs: 1,
        batchSize: 1,
        gradientAccumulationSteps: 4,
        maxLength: 1024,
        dtype: "bfloat16",
        loadIn4bit: true,
      },
    },
  ],
};

const initialDatasets = {
  datasets: [],
};

const initialJobs = {
  jobs: [],
};

const initialModels = {
  models: [],
};

const topNavItems = [
  { id: "overview", label: "Overview" },
  { id: "training", label: "Training" },
  { id: "models", label: "Models" },
  { id: "evaluation", label: "Evaluation" },
  { id: "datasets", label: "Datasets" },
];

const sidebarSections = [
  {
    title: "Platform",
    items: [
      { id: "overview", label: "Overview", dot: "train" },
      { id: "training", label: "Run builder", dot: "live" },
      { id: "models", label: "Model registry", dot: "live" },
    ],
  },
  {
    title: "Review",
    items: [
      { id: "evaluation", label: "Evaluation reports", dot: "eval" },
      { id: "datasets", label: "Dataset registry", dot: "idle" },
    ],
  },
];

function App() {
  const [user, setUser] = useState(null);
  const [authToken, setAuthToken] = useState(() => localStorage.getItem(AUTH_TOKEN_KEY) ?? "");
  const [authMode, setAuthMode] = useState("login");
  const [authError, setAuthError] = useState("");
  const [authWorking, setAuthWorking] = useState(false);
  const [authForm, setAuthForm] = useState({ name: "", email: "", password: "" });
  const [activeScreen, setActiveScreen] = useState("overview");
  const [apiStatus, setApiStatus] = useState("connecting");
  const [isBootLoading, setIsBootLoading] = useState(true);
  const [dashboard, setDashboard] = useState(initialDashboard);
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(initialDashboard.defaultRunId);
  const [summary, setSummary] = useState(initialSummary);
  const [options, setOptions] = useState(initialOptions);
  const [datasetsResponse, setDatasetsResponse] = useState(initialDatasets);
  const [jobsResponse, setJobsResponse] = useState(initialJobs);
  const [modelsResponse, setModelsResponse] = useState(initialModels);
  const [bundles, setBundles] = useState([]);
  const [selectedBundleId, setSelectedBundleId] = useState(null);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [datasetUpload, setDatasetUpload] = useState({ name: "", kind: "qa", files: [] });
  const [datasetFormMode, setDatasetFormMode] = useState("upload");
  const [datasetGeneration, setDatasetGeneration] = useState({
    name: "",
    kind: "qa",
    files: [],
    corpusDatasetId: "",
    modelName: "openrouter/auto",
    apiBase: "https://openrouter.ai/api/v1",
    qaNumPairs: 4,
    qaChunkSize: 2500,
    qaChunkOverlap: 150,
    corpusChunkSize: 1000,
    corpusChunkOverlap: 150,
    ingestToPgvector: true,
    collectionName: "",
  });
  const [uploadError, setUploadError] = useState("");
  const [uploadingDataset, setUploadingDataset] = useState(false);
  const [generationError, setGenerationError] = useState("");
  const [generatingDataset, setGeneratingDataset] = useState(false);
  const [deletingDatasetId, setDeletingDatasetId] = useState("");
  const [ingestingDatasetId, setIngestingDatasetId] = useState("");
  const [downloadingDatasetId, setDownloadingDatasetId] = useState("");
  const [modelError, setModelError] = useState("");
  const [importingModel, setImportingModel] = useState(false);
  const [downloadingModelId, setDownloadingModelId] = useState("");
  const [deletingModelId, setDeletingModelId] = useState("");
  const [bundleError, setBundleError] = useState("");
  const [creatingBundle, setCreatingBundle] = useState(false);
  const [bundleSuccess, setBundleSuccess] = useState("");
  const [toasts, setToasts] = useState([]);
  const [launchingBundleId, setLaunchingBundleId] = useState("");
  const [downloadingBundleId, setDownloadingBundleId] = useState("");
  const [deletingBundleId, setDeletingBundleId] = useState("");
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationNotice, setEvaluationNotice] = useState("");
  const [creatingEvaluationJob, setCreatingEvaluationJob] = useState(false);
  const [downloadingRunId, setDownloadingRunId] = useState("");
  const [deletingRunId, setDeletingRunId] = useState("");
  const [trainingForm, setTrainingForm] = useState({
    runName: "",
    executionMode: "colab",
    baseModel: initialOptions.models[0].id,
    peftMethod: "qlora",
    loraRank: 16,
    qaDatasetId: "",
    instructionDatasetId: "",
    configMode: "preset",
    presetId: "balanced-qlora",
    manualConfig: { ...initialOptions.configPresets[0].manualDefaults },
    uploadedConfigText: "",
  });
  const [evaluationForm, setEvaluationForm] = useState({
    runName: "",
    benchmarkMode: "existing",
    benchmarkDatasetId: "",
    corpusDatasetId: "",
    modelId: "",
    runBase: true,
    runRag: true,
    runFineTuned: true,
    runFineTunedRag: true,
  });
  const [modelImportForm, setModelImportForm] = useState({
    name: "",
    source: "colab",
    baseModel: initialOptions.models[0].id,
    peftMethod: "qlora",
    loraRank: 16,
    files: [],
  });
  const previousJobsRef = useRef(new Map());

  useEffect(() => {
    bootstrapSession();
  }, []);

  useEffect(() => {
    if (user && selectedRunId) {
      loadRunData(selectedRunId);
    }
  }, [selectedRunId, user]);

  useEffect(() => {
    if (!user) {
      return undefined;
    }
    const currentJobs = jobsResponse.jobs ?? [];
    const hasRunningJobs = currentJobs.some((job) => ["queued", "running"].includes(job.status));
    if (!hasRunningJobs) {
      return undefined;
    }
    const intervalId = window.setInterval(() => {
      refreshJobsOnly();
      refreshDatasetsAndBundles();
    }, 4000);
    return () => window.clearInterval(intervalId);
  }, [jobsResponse, user]);

  useEffect(() => {
    if (options.models?.length && !trainingForm.baseModel) {
      setTrainingForm((current) => ({ ...current, baseModel: options.models[0].id }));
    }
  }, [options.models, trainingForm.baseModel]);

  useEffect(() => {
    if (options.models?.length && !modelImportForm.baseModel) {
      setModelImportForm((current) => ({ ...current, baseModel: options.models[0].id }));
    }
  }, [options.models, modelImportForm.baseModel]);

  useEffect(() => {
    const allDatasets = datasetsResponse.datasets ?? [];
    setEvaluationForm((current) => ({
      ...current,
      benchmarkDatasetId:
        current.benchmarkDatasetId && allDatasets.some((dataset) => dataset.id === current.benchmarkDatasetId)
          ? current.benchmarkDatasetId
          : "",
      corpusDatasetId:
        current.corpusDatasetId && allDatasets.some((dataset) => dataset.id === current.corpusDatasetId)
          ? current.corpusDatasetId
          : "",
    }));
  }, [datasetsResponse]);

  useEffect(() => {
    const availableModels = (modelsResponse.models ?? []).filter(
      (model) => (model.storageProvider ?? "azure_blob") === "azure_blob" && model.archivePath
    );
    setEvaluationForm((current) => ({
      ...current,
      modelId:
        current.modelId && availableModels.some((model) => model.id === current.modelId)
          ? current.modelId
          : availableModels[0]?.id ?? "",
    }));
  }, [modelsResponse]);

  useEffect(() => {
    const previousJobs = previousJobsRef.current;
    for (const job of jobsResponse.jobs ?? []) {
      const previousSnapshot = previousJobs.get(job.id);
      const previousStatus = previousSnapshot?.status;
      if (previousStatus && previousStatus !== job.status) {
        if (job.status === "completed") {
          pushToast("success", `${job.title} completed successfully.`);
        } else if (job.status === "failed") {
          pushToast("error", job.error ? `${job.title} failed: ${job.error}` : `${job.title} failed.`);
        }
      }
      if (job.type === "evaluation") {
        const previousCompletedModes = previousSnapshot?.completedModes ?? [];
        const currentCompletedModes = job.metadata?.completedModes ?? [];
        const newlyCompletedModes = currentCompletedModes.filter((mode) => !previousCompletedModes.includes(mode));
        for (const mode of newlyCompletedModes) {
          pushToast("success", `${job.title}: ${prettifyLabel(mode.replaceAll("_", " "))} finished.`);
        }
      }
      previousJobs.set(job.id, {
        status: job.status,
        completedModes: job.metadata?.completedModes ?? [],
      });
    }

    for (const existingId of Array.from(previousJobs.keys())) {
      if (!(jobsResponse.jobs ?? []).some((job) => job.id === existingId)) {
        previousJobs.delete(existingId);
      }
    }
  }, [jobsResponse]);

  const datasets = datasetsResponse.datasets ?? [];
  const jobs = jobsResponse.jobs ?? [];
  const models = modelsResponse.models ?? [];
  const deployableEvaluationModels = models.filter(
    (model) => (model.storageProvider ?? "azure_blob") === "azure_blob" && model.archivePath
  );
  const selectedBundle = useMemo(
    () => bundles.find((bundle) => bundle.id === selectedBundleId) ?? bundles[0] ?? null,
    [bundles, selectedBundleId]
  );
  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? models[0] ?? null,
    [models, selectedModelId]
  );
  const selectedJob = useMemo(
    () => jobs.find((job) => job.id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId]
  );
  const runningJobs = useMemo(
    () => jobs.filter((job) => ["queued", "running"].includes(job.status)),
    [jobs]
  );
  const evaluationJobs = useMemo(
    () => jobs.filter((job) => job.type === "evaluation"),
    [jobs]
  );
  const runningDatasetJobs = useMemo(
    () => jobs.filter((job) => ["queued", "running"].includes(job.status) && ["dataset-generation", "vector-ingestion"].includes(job.type)),
    [jobs]
  );

  const selectedRun = useMemo(
    () => dashboard.recentRuns.find((run) => run.id === selectedRunId) ?? dashboard.recentRuns[0] ?? null,
    [dashboard.recentRuns, selectedRunId]
  );

  const primaryModel = options.models?.[0] ?? initialOptions.models[0];

  const qaDatasets = datasets.filter((dataset) => dataset.kind === "qa");
  const instructionDatasets = datasets.filter((dataset) => dataset.kind === "instruction");
  const benchmarkDatasets = datasets.filter((dataset) => dataset.kind === "benchmark");
  const corpusDatasets = datasets.filter((dataset) => dataset.kind === "corpus");
  const uploadSummaryText = datasetUpload.files.length
    ? `${datasetUpload.files.length} item${datasetUpload.files.length > 1 ? "s" : ""} selected`
    : "No files selected yet";
  const generationSummaryText = datasetGeneration.files.length
    ? `${datasetGeneration.files.length} item${datasetGeneration.files.length > 1 ? "s" : ""} selected`
    : "No source documents selected yet";
  const modelImportSummaryText = modelImportForm.files.length
    ? `${modelImportForm.files.length} item${modelImportForm.files.length > 1 ? "s" : ""} selected`
    : "No model files selected yet";
  const canPrepareRun = Boolean(trainingForm.qaDatasetId && trainingForm.instructionDatasetId);

  function pushToast(tone, text) {
    const toast = { id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, tone, text };
    setToasts((current) => [...current, toast]);
    if (tone !== "error") {
      window.setTimeout(() => {
        setToasts((current) => current.filter((item) => item.id !== toast.id));
      }, 4500);
    }
  }

  function dismissToast(id) {
    setToasts((current) => current.filter((item) => item.id !== id));
  }

  function clearAuthState() {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    setAuthToken("");
    setUser(null);
    setApiStatus("offline");
  }

  async function authFetch(url, options = {}, activeToken = authToken) {
    const headers = new Headers(options.headers ?? {});
    if (activeToken) {
      headers.set("Authorization", `Bearer ${activeToken}`);
    }
    if (options.body && !(options.body instanceof FormData) && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    const response = await fetch(url, { ...options, headers });
    if (response.status === 401) {
      clearAuthState();
      throw new Error("Your session expired. Please sign in again.");
    }
    return response;
  }

  async function bootstrapSession() {
    const storedToken = localStorage.getItem(AUTH_TOKEN_KEY);
    if (!storedToken) {
      setIsBootLoading(false);
      setApiStatus("offline");
      return;
    }

    setAuthToken(storedToken);
    try {
      const response = await fetch("/api/auth/me", {
        headers: { Authorization: `Bearer ${storedToken}` },
      });
      if (!response.ok) {
        throw new Error("Stored session is no longer valid.");
      }
      const payload = await response.json();
      setUser(payload.user);
      await loadInitialData(storedToken);
    } catch (error) {
      clearAuthState();
      setAuthError("Your saved session could not be restored. Please sign in again.");
      setIsBootLoading(false);
    }
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    setAuthError("");

    if (!authForm.email.trim() || !authForm.password.trim()) {
      setAuthError("Email and password are required.");
      return;
    }
    if (!authForm.email.includes("@")) {
      setAuthError("Please provide a valid email address.");
      return;
    }
    if (authForm.password.length < 8) {
      setAuthError("Password must be at least 8 characters long.");
      return;
    }
    if (authMode === "register" && authForm.name.trim().length < 2) {
      setAuthError("Name must be at least 2 characters long.");
      return;
    }

    setAuthWorking(true);
    setIsBootLoading(true);
    try {
      const endpoint = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
      const payload =
        authMode === "register"
          ? {
              name: authForm.name.trim(),
              email: authForm.email.trim(),
              password: authForm.password,
            }
          : {
              email: authForm.email.trim(),
              password: authForm.password,
            };

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const authPayload = await response.json();
      localStorage.setItem(AUTH_TOKEN_KEY, authPayload.token);
      setAuthToken(authPayload.token);
      setUser(authPayload.user);
      setAuthForm((current) => ({ ...current, password: "" }));
      await loadInitialData(authPayload.token);
    } catch (error) {
      setAuthError(error.message || "Authentication failed.");
      setIsBootLoading(false);
    } finally {
      setAuthWorking(false);
    }
  }

  async function handleLogout() {
    try {
      await authFetch("/api/auth/logout", { method: "POST" });
    } catch (error) {
      // Clear local auth state even if the backend session is already gone.
    } finally {
      clearAuthState();
      setSummary(initialSummary);
      setBundles([]);
      setSelectedBundleId(null);
      setJobsResponse(initialJobs);
      setSelectedJobId(null);
      setModelsResponse(initialModels);
      setSelectedModelId(null);
      setRuns([]);
    }
  }

  async function loadInitialData(activeToken = authToken) {
    try {
      const [dashboardResponse, runsResponse, optionsResponse] = await Promise.all([
        authFetch("/api/dashboard", {}, activeToken),
        authFetch("/api/runs", {}, activeToken),
        authFetch("/api/platform/options", {}, activeToken),
      ]);

      if (!dashboardResponse.ok || !runsResponse.ok || !optionsResponse.ok) {
        throw new Error("API unavailable");
      }

      const dashboardJson = await dashboardResponse.json();
      const runsJson = await runsResponse.json();
      const optionsJson = await optionsResponse.json();

      setDashboard(dashboardJson);
      setRuns(runsJson.runs ?? []);
      setOptions(optionsJson);
      setSelectedRunId(dashboardJson.defaultRunId ?? runsJson.runs?.[0]?.id ?? initialDashboard.defaultRunId);
      setApiStatus("connected");

      setTrainingForm((current) => {
        const selectedPreset =
          optionsJson.configPresets.find((preset) => preset.id === current.presetId) ?? optionsJson.configPresets[0];
        return {
          ...current,
          baseModel: current.baseModel || optionsJson.models[0]?.id || initialOptions.models[0].id,
          executionMode: current.executionMode || "colab",
          presetId: selectedPreset?.id ?? current.presetId,
          manualConfig: selectedPreset?.manualDefaults ?? current.manualConfig,
        };
      });
      setIsBootLoading(false);

      Promise.allSettled([
        authFetch("/api/datasets", {}, activeToken),
        authFetch("/api/run-bundles", {}, activeToken),
        authFetch("/api/jobs", {}, activeToken),
        authFetch("/api/models", {}, activeToken),
      ]).then(async ([datasetsResult, bundlesResult, jobsResult, modelsResult]) => {
        if (datasetsResult.status === "fulfilled" && datasetsResult.value.ok) {
          const datasetsJson = await datasetsResult.value.json();
          setDatasetsResponse(datasetsJson);
          setTrainingForm((current) => ({
            ...current,
            qaDatasetId: current.qaDatasetId && datasetsJson.datasets.some((dataset) => dataset.id === current.qaDatasetId)
              ? current.qaDatasetId
              : "",
            instructionDatasetId:
              current.instructionDatasetId && datasetsJson.datasets.some((dataset) => dataset.id === current.instructionDatasetId)
                ? current.instructionDatasetId
                : "",
          }));
        }
        if (bundlesResult.status === "fulfilled" && bundlesResult.value.ok) {
          const bundlesJson = await bundlesResult.value.json();
          setBundles(bundlesJson.bundles ?? []);
          setSelectedBundleId(bundlesJson.bundles?.[0]?.id ?? null);
        }
        if (jobsResult.status === "fulfilled" && jobsResult.value.ok) {
          const jobsJson = await jobsResult.value.json();
          setJobsResponse(jobsJson);
          setSelectedJobId(jobsJson.jobs?.[0]?.id ?? null);
        }
        if (modelsResult.status === "fulfilled" && modelsResult.value.ok) {
          const modelsJson = await modelsResult.value.json();
          setModelsResponse(modelsJson);
          setSelectedModelId(modelsJson.models?.[0]?.id ?? null);
        }
      });
    } catch (error) {
      setDashboard(initialDashboard);
      setRuns([]);
      setSummary(initialSummary);
      setModelsResponse(initialModels);
      setApiStatus("offline");
      setSelectedBundleId(null);
      setSelectedModelId(null);
    }
    finally {
      setIsBootLoading(false);
    }
  }

  async function loadRunData(runId, activeToken = authToken) {
    try {
      const summaryResponse = await authFetch(`/api/runs/${runId}/summary`, {}, activeToken);
      if (!summaryResponse.ok) {
        throw new Error("Run data unavailable");
      }
      const summaryJson = await summaryResponse.json();
      setSummary(summaryJson);
    } catch (error) {
      setSummary(initialSummary);
    }
  }

  async function refreshDatasetsAndBundles(activeToken = authToken) {
    try {
      const [datasetsApiResponse, bundlesResponse, dashboardResponse, jobsApiResponse, runsResponse, modelsApiResponse] = await Promise.all([
        authFetch("/api/datasets", {}, activeToken),
        authFetch("/api/run-bundles", {}, activeToken),
        authFetch("/api/dashboard", {}, activeToken),
        authFetch("/api/jobs", {}, activeToken),
        authFetch("/api/runs", {}, activeToken),
        authFetch("/api/models", {}, activeToken),
      ]);
      if (datasetsApiResponse.ok) {
        setDatasetsResponse(await datasetsApiResponse.json());
      }
      if (bundlesResponse.ok) {
        const bundlesJson = await bundlesResponse.json();
        setBundles(bundlesJson.bundles ?? []);
        setSelectedBundleId((current) => current ?? bundlesJson.bundles?.[0]?.id ?? null);
      }
      if (dashboardResponse.ok) {
        setDashboard(await dashboardResponse.json());
      }
      if (runsResponse.ok) {
        const runsJson = await runsResponse.json();
        setRuns(runsJson.runs ?? []);
        setSelectedRunId((current) =>
          current && (runsJson.runs ?? []).some((run) => run.id === current)
            ? current
            : runsJson.runs?.[0]?.id ?? null
        );
      }
      if (jobsApiResponse.ok) {
        const jobsJson = await jobsApiResponse.json();
        setJobsResponse(jobsJson);
        setSelectedJobId((current) => current ?? jobsJson.jobs?.[0]?.id ?? null);
      }
      if (modelsApiResponse.ok) {
        const modelsJson = await modelsApiResponse.json();
        setModelsResponse(modelsJson);
        setSelectedModelId((current) => current ?? modelsJson.models?.[0]?.id ?? null);
      }
    } catch (error) {
      // Keep current state if refresh fails.
    }
  }

  async function refreshJobsOnly(activeToken = authToken) {
    try {
      const response = await authFetch("/api/jobs", {}, activeToken);
      if (!response.ok) {
        return;
      }
      const jobsJson = await response.json();
      setJobsResponse(jobsJson);
      setSelectedJobId((current) => current ?? jobsJson.jobs?.[0]?.id ?? null);
    } catch (error) {
      // Ignore transient job refresh failures.
    }
  }

  function updateTrainingField(field, value) {
    setTrainingForm((current) => ({ ...current, [field]: value }));
  }

  function updateManualConfig(field, value) {
    setTrainingForm((current) => ({
      ...current,
      manualConfig: {
        ...current.manualConfig,
        [field]: value,
      },
    }));
  }

  function handlePresetChange(presetId) {
    const preset = options.configPresets.find((entry) => entry.id === presetId);
    setTrainingForm((current) => ({
      ...current,
      presetId,
      manualConfig: preset?.manualDefaults ?? current.manualConfig,
    }));
  }

  async function handleDatasetUpload(event) {
    event.preventDefault();
    setUploadError("");
    setBundleSuccess("");

    if (!datasetUpload.name.trim()) {
      setUploadError("Give the dataset a clear name first.");
      return;
    }
    if (datasets.some((dataset) => normalizedClientName(dataset.name) === normalizedClientName(datasetUpload.name))) {
      setUploadError(`A dataset named "${datasetUpload.name.trim()}" already exists in this account.`);
      return;
    }
    if (!datasetUpload.files.length) {
      setUploadError("Choose at least one dataset file to upload.");
      return;
    }

    setUploadingDataset(true);
    try {
      const filePayloads = await Promise.all(datasetUpload.files.map((entry) => buildUploadFilePayload(entry)));

      const response = await authFetch("/api/datasets/upload", {
        method: "POST",
        body: JSON.stringify({
          name: datasetUpload.name,
          kind: datasetUpload.kind,
          files: filePayloads,
        }),
      });

      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const payload = await response.json();
      const uploadedDataset = payload.dataset;
      setDatasetUpload({ name: "", kind: "qa", files: [] });
      setBundleSuccess(
        uploadedDataset.kind === "benchmark"
          ? `Uploaded ${uploadedDataset.name}. It is now available in evaluation.`
          : `Uploaded ${uploadedDataset.name}. It is now available in the run builder.`
      );
      await refreshDatasetsAndBundles();

      if (uploadedDataset.kind === "qa") {
        updateTrainingField("qaDatasetId", uploadedDataset.id);
      } else if (uploadedDataset.kind === "instruction") {
        updateTrainingField("instructionDatasetId", uploadedDataset.id);
      }
    } catch (error) {
      setUploadError(error.message || "Dataset upload failed.");
    } finally {
      setUploadingDataset(false);
    }
  }

  async function handleCreateBundle(event) {
    event.preventDefault();
    setBundleError("");
    setBundleSuccess("");

    if (!trainingForm.runName.trim()) {
      setBundleError("Trial name is required before preparing a bundle.");
      return;
    }
    if (bundles.some((bundle) => normalizedClientName(bundle.runName) === normalizedClientName(trainingForm.runName))) {
      setBundleError(`A prepared run named "${trainingForm.runName.trim()}" already exists in this account.`);
      return;
    }
    if (!trainingForm.qaDatasetId || !trainingForm.instructionDatasetId) {
      setBundleError("Pick both a QA dataset and an instruction dataset before launching.");
      return;
    }

    setCreatingBundle(true);
    try {
      const response = await authFetch("/api/run-bundles", {
        method: "POST",
        body: JSON.stringify({
          runName: trainingForm.runName,
          executionMode: trainingForm.executionMode,
          baseModel: trainingForm.baseModel,
          peftMethod: trainingForm.peftMethod,
          loraRank: Number(trainingForm.loraRank),
          qaDatasetId: trainingForm.qaDatasetId,
          instructionDatasetId: trainingForm.instructionDatasetId,
          configMode: trainingForm.configMode,
          presetId: trainingForm.presetId,
          manualConfig: trainingForm.configMode === "manual" ? trainingForm.manualConfig : null,
          uploadedConfigText: trainingForm.configMode === "upload" ? trainingForm.uploadedConfigText : null,
        }),
      });

      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const payload = await response.json();
      setBundleSuccess(`Prepared bundle ${payload.bundle.runName}. You can now download it or follow the launch instructions.`);
      setSelectedBundleId(payload.bundle.id);
      await refreshDatasetsAndBundles();
    } catch (error) {
      setBundleError(error.message || "Bundle creation failed.");
    } finally {
      setCreatingBundle(false);
    }
  }

  async function handleDeleteDataset(dataset) {
    const confirmed = window.confirm(`Remove "${dataset.name}" from your dataset registry?`);
    if (!confirmed) {
      return;
    }

    setUploadError("");
    setBundleSuccess("");
    setDeletingDatasetId(dataset.id);
    try {
      const response = await authFetch(`/api/datasets/${dataset.id}`, { method: "DELETE" });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      setTrainingForm((current) => ({
        ...current,
        qaDatasetId: current.qaDatasetId === dataset.id ? "" : current.qaDatasetId,
        instructionDatasetId: current.instructionDatasetId === dataset.id ? "" : current.instructionDatasetId,
      }));
      setBundleSuccess(`Removed ${dataset.name} from this account.`);
      await refreshDatasetsAndBundles();
    } catch (error) {
      setUploadError(error.message || "Dataset removal failed.");
    } finally {
      setDeletingDatasetId("");
    }
  }

  async function handleDownloadDataset(dataset) {
    setUploadError("");
    setGenerationError("");
    setDownloadingDatasetId(dataset.id);
    try {
      const response = await authFetch(`/api/datasets/${dataset.id}/download`);
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = inferDatasetDownloadName(dataset);
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setUploadError(error.message || "Dataset download failed.");
    } finally {
      setDownloadingDatasetId("");
    }
  }

  async function handleGenerateDataset(event) {
    event.preventDefault();
    setGenerationError("");
    setBundleSuccess("");

    if (!datasetGeneration.name.trim()) {
      setGenerationError("Give the generated dataset a clear name first.");
      return;
    }
    if (datasets.some((dataset) => normalizedClientName(dataset.name) === normalizedClientName(datasetGeneration.name))) {
      setGenerationError(`A dataset named "${datasetGeneration.name.trim()}" already exists in this account.`);
      return;
    }
    if (datasetGeneration.kind === "benchmark" && !datasetGeneration.corpusDatasetId) {
      setGenerationError("Choose a corpus dataset before generating a benchmark dataset.");
      return;
    }
    if (datasetGeneration.kind !== "benchmark" && !datasetGeneration.files.length) {
      setGenerationError("Choose source documents before generating a dataset.");
      return;
    }

    setGeneratingDataset(true);
    try {
      const filePayloads =
        datasetGeneration.kind === "benchmark"
          ? []
          : await Promise.all(datasetGeneration.files.map((entry) => buildUploadFilePayload(entry)));

      const response = await authFetch("/api/datasets/generate", {
        method: "POST",
        body: JSON.stringify({
          name: datasetGeneration.name,
          kind: datasetGeneration.kind,
          files: filePayloads,
          corpusDatasetId: datasetGeneration.kind === "benchmark" ? datasetGeneration.corpusDatasetId : null,
          modelName: datasetGeneration.modelName,
          apiBase: datasetGeneration.apiBase,
          chunkSize: Number(datasetGeneration.corpusChunkSize),
          chunkOverlap: Number(datasetGeneration.corpusChunkOverlap),
          qaNumPairs: Number(datasetGeneration.qaNumPairs),
          qaChunkSize: Number(datasetGeneration.qaChunkSize),
          qaChunkOverlap: Number(datasetGeneration.qaChunkOverlap),
          ingestToPgvector: datasetGeneration.kind === "corpus" ? datasetGeneration.ingestToPgvector : false,
          collectionName:
            datasetGeneration.kind === "corpus" && datasetGeneration.ingestToPgvector
              ? datasetGeneration.collectionName.trim() || slugifyValue(datasetGeneration.name)
              : null,
        }),
      });

      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const payload = await response.json();
      const createdJob = payload.job;
      setDatasetGeneration({
        name: "",
        kind: "qa",
        files: [],
        corpusDatasetId: "",
        modelName: "openrouter/auto",
        apiBase: "https://openrouter.ai/api/v1",
        qaNumPairs: 4,
        qaChunkSize: 2500,
        qaChunkOverlap: 150,
        corpusChunkSize: 1000,
        corpusChunkOverlap: 150,
        ingestToPgvector: true,
        collectionName: "",
      });
      setBundleSuccess(`Queued dataset generation job for ${datasetGeneration.name}. DALTP will track it in the jobs panel.`);
      setJobsResponse((current) => ({ jobs: [createdJob, ...(current.jobs ?? [])] }));
      setSelectedJobId(createdJob.id);
      await refreshJobsOnly();
    } catch (error) {
      setGenerationError(error.message || "Dataset generation failed.");
    } finally {
      setGeneratingDataset(false);
    }
  }

  function normalizeSelectedFiles(fileList) {
    return {
      files: Array.from(fileList ?? []).map((file) => ({
        file,
        displayName: file.webkitRelativePath || file.name,
        relativePath: file.webkitRelativePath || file.name,
      })),
    };
  }

  function normalizeModelImportFiles(fileList) {
    const selected = Array.from(fileList ?? []).map((file) => ({
      file,
      displayName: file.webkitRelativePath || file.name,
      relativePath: file.webkitRelativePath || file.name,
    }));

    let skippedCheckpoints = 0;
    const files = selected.filter((entry) => {
      const relativePath = String(entry.relativePath || "").replace(/\\/g, "/");
      if (!relativePath.includes("/")) {
        return true;
      }
      const pathParts = relativePath.split("/").filter(Boolean);
      const hasCheckpointDir = pathParts.some((part) => /^checkpoint-\d+$/i.test(part));
      if (hasCheckpointDir) {
        skippedCheckpoints += 1;
        return false;
      }
      return true;
    });

    return { files, skippedCheckpoints };
  }

  function handleDatasetFileSelection(fileList) {
    const normalized = normalizeSelectedFiles(fileList);
    setDatasetUpload((current) => ({
      ...current,
      ...normalized,
    }));
  }

  function removeDatasetUploadFile(relativePath) {
    setDatasetUpload((current) => ({
      ...current,
      files: current.files.filter((entry) => entry.relativePath !== relativePath),
    }));
  }

  function handleGenerationFileSelection(fileList) {
    const normalized = normalizeSelectedFiles(fileList);
    setDatasetGeneration((current) => ({
      ...current,
      ...normalized,
    }));
  }

  function removeGenerationFile(relativePath) {
    setDatasetGeneration((current) => ({
      ...current,
      files: current.files.filter((entry) => entry.relativePath !== relativePath),
    }));
  }

  function handleModelImportFileSelection(fileList) {
    const normalized = normalizeModelImportFiles(fileList);
    setModelImportForm((current) => ({
      ...current,
      files: normalized.files,
    }));
    if (normalized.skippedCheckpoints) {
      pushToast(
        "info",
        `Skipped ${normalized.skippedCheckpoints} checkpoint file${normalized.skippedCheckpoints > 1 ? "s" : ""} during model import. DALTP only needs the final exported artifact files.`
      );
    }
  }

  async function handleIngestCorpusDataset(dataset) {
    const suggestedName = dataset.vectorStore?.collectionName || slugifyValue(dataset.name);
    const chosenCollection = window.prompt("pgvector namespace", suggestedName);
    if (!chosenCollection || !chosenCollection.trim()) {
      return;
    }

    setGenerationError("");
    setUploadError("");
    setBundleSuccess("");
    setIngestingDatasetId(dataset.id);
    try {
      const response = await authFetch(`/api/datasets/${dataset.id}/ingest-pgvector`, {
        method: "POST",
        body: JSON.stringify({
          collectionName: chosenCollection.trim(),
          chunkSize: 1000,
          chunkOverlap: 150,
        }),
      });

      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const payload = await response.json();
      setBundleSuccess(`Queued pgvector ingestion for ${dataset.name}. DALTP will update the registry when the job finishes.`);
      setJobsResponse((current) => ({ jobs: [payload.job, ...(current.jobs ?? [])] }));
      setSelectedJobId(payload.job.id);
      await refreshJobsOnly();
    } catch (error) {
      setGenerationError(error.message || "pgvector ingestion failed.");
    } finally {
      setIngestingDatasetId("");
    }
  }

  async function handleImportModelArtifact(event) {
    event.preventDefault();
    setModelError("");
    setBundleSuccess("");

    if (!modelImportForm.name.trim()) {
      setModelError("Give the trained model artifact a clear name first.");
      return;
    }
    if (models.some((model) => normalizedClientName(model.name) === normalizedClientName(modelImportForm.name))) {
      setModelError(`A model artifact named "${modelImportForm.name.trim()}" already exists in this account.`);
      return;
    }
    if (!modelImportForm.files.length) {
      setModelError("Choose the trained model files before importing.");
      return;
    }

    setImportingModel(true);
    try {
      const singleZipEntry =
        modelImportForm.files.length === 1 && /\.zip$/i.test(modelImportForm.files[0]?.file?.name || "");

      let response;
      if (singleZipEntry) {
        const formData = new FormData();
        formData.append("name", modelImportForm.name);
        formData.append("source", modelImportForm.source);
        formData.append("baseModel", modelImportForm.baseModel);
        formData.append("peftMethod", modelImportForm.peftMethod);
        formData.append("loraRank", String(Number(modelImportForm.loraRank)));
        formData.append("archive", modelImportForm.files[0].file);
        response = await authFetch("/api/models/import-archive", {
          method: "POST",
          body: formData,
        });
      } else {
        const filePayloads = await Promise.all(modelImportForm.files.map((entry) => buildUploadFilePayload(entry)));
        response = await authFetch("/api/models/import", {
          method: "POST",
          body: JSON.stringify({
            name: modelImportForm.name,
            source: modelImportForm.source,
            baseModel: modelImportForm.baseModel,
            peftMethod: modelImportForm.peftMethod,
            loraRank: Number(modelImportForm.loraRank),
            files: filePayloads,
          }),
        });
      }
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      const payload = await response.json();
      setBundleSuccess(`Imported model artifact ${payload.model.name}. It is now available in your registry.`);
      setModelImportForm({
        name: "",
        source: "colab",
        baseModel: options.models?.[0]?.id ?? initialOptions.models[0].id,
        peftMethod: "qlora",
        loraRank: 16,
        files: [],
      });
      await refreshDatasetsAndBundles();
    } catch (error) {
      setModelError(error.message || "Model import failed.");
    } finally {
      setImportingModel(false);
    }
  }

  async function handleDownloadModel(model) {
    setModelError("");
    setDownloadingModelId(model.id);
    try {
      const response = await authFetch(`/api/models/${model.id}/download`);
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `${slugifyValue(model.name)}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setModelError(error.message || "Model download failed.");
    } finally {
      setDownloadingModelId("");
    }
  }

  async function handleLaunchLocalTraining(bundle) {
    setBundleError("");
    setBundleSuccess("");
    setLaunchingBundleId(bundle.id);
    try {
      const response = await authFetch(`/api/run-bundles/${bundle.id}/launch-local`, {
        method: "POST",
      });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      const payload = await response.json();
      setBundleSuccess(`Started local training for ${bundle.runName}. Progress is now tracked in Jobs.`);
      setJobsResponse((current) => ({ jobs: [payload.job, ...(current.jobs ?? [])] }));
      setSelectedJobId(payload.job.id);
      await refreshJobsOnly();
    } catch (error) {
      setBundleError(error.message || "Local training launch failed.");
    } finally {
      setLaunchingBundleId("");
    }
  }

  async function handleCreateEvaluationJob(event) {
    event.preventDefault();
    setEvaluationError("");
    setEvaluationNotice("");

    if (!evaluationForm.runName.trim()) {
      setEvaluationError("Evaluation trial name is required.");
      return;
    }
    if (runs.some((run) => normalizedClientName(run.name) === normalizedClientName(evaluationForm.runName))) {
      setEvaluationError(`An evaluation run named "${evaluationForm.runName.trim()}" already exists.`);
      return;
    }
    if (evaluationForm.benchmarkMode === "existing" && !evaluationForm.benchmarkDatasetId) {
      setEvaluationError("Choose a benchmark dataset before running evaluation.");
      return;
    }
    if ((evaluationForm.runRag || evaluationForm.runFineTunedRag) && !evaluationForm.corpusDatasetId) {
      setEvaluationError("Choose a corpus dataset for RAG evaluation.");
      return;
    }
    if ((evaluationForm.runFineTuned || evaluationForm.runFineTunedRag) && !evaluationForm.modelId) {
      setEvaluationError("Choose a model artifact for fine-tuned evaluation.");
      return;
    }

    setCreatingEvaluationJob(true);
    try {
      const response = await authFetch("/api/evaluation/jobs", {
        method: "POST",
        body: JSON.stringify(evaluationForm),
      });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      const payload = await response.json();
      setEvaluationNotice(`Queued evaluation job ${evaluationForm.runName}. You can keep working while DALTP scores the selected systems.`);
      setJobsResponse((current) => ({ jobs: [payload.job, ...(current.jobs ?? [])] }));
      setSelectedJobId(payload.job.id);
      await refreshJobsOnly();
    } catch (error) {
      setEvaluationError(error.message || "Evaluation launch failed.");
    } finally {
      setCreatingEvaluationJob(false);
    }
  }

  async function handleDownloadEvaluationRun(runId, runName) {
    setEvaluationError("");
    setDownloadingRunId(runId);
    try {
      const response = await authFetch(`/api/runs/${runId}/download`);
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `${slugifyValue(runName || "evaluation-run")}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setEvaluationError(error.message || "Evaluation download failed.");
    } finally {
      setDownloadingRunId("");
    }
  }

  async function handleDeleteEvaluationRun(runId, runName) {
    const confirmed = window.confirm(`Remove the evaluation run "${runName}"?`);
    if (!confirmed) {
      return;
    }

    setEvaluationError("");
    setEvaluationNotice("");
    setDeletingRunId(runId);
    try {
      const response = await authFetch(`/api/runs/${runId}`, { method: "DELETE" });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      if (selectedRunId === runId) {
        setSelectedRunId(null);
        setSummary(initialSummary);
      }
      setEvaluationNotice(`Removed evaluation run ${runName}.`);
      await refreshDatasetsAndBundles();
    } catch (error) {
      setEvaluationError(error.message || "Evaluation removal failed.");
    } finally {
      setDeletingRunId("");
    }
  }

  async function handleDeleteModel(model) {
    const confirmed = window.confirm(`Remove the model artifact "${model.name}"?`);
    if (!confirmed) {
      return;
    }

    setModelError("");
    setDeletingModelId(model.id);
    try {
      const response = await authFetch(`/api/models/${model.id}`, { method: "DELETE" });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      setBundleSuccess(`Removed model artifact ${model.name}.`);
      setSelectedModelId((current) => (current === model.id ? null : current));
      await refreshDatasetsAndBundles();
    } catch (error) {
      setModelError(error.message || "Model removal failed.");
    } finally {
      setDeletingModelId("");
    }
  }

  async function handleDownloadBundle(bundle) {
    setBundleError("");
    setDownloadingBundleId(bundle.id);
    try {
      const response = await authFetch(bundle.downloadUrl);
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }

      const blob = await response.blob();
      const objectUrl = window.URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `${slugifyValue(bundle.runName || "run-bundle")}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.URL.revokeObjectURL(objectUrl);
    } catch (error) {
      setBundleError(error.message || "Bundle download failed.");
    } finally {
      setDownloadingBundleId("");
    }
  }

  async function handleDeleteBundle(bundle) {
    const confirmed = window.confirm(`Remove the run bundle "${bundle.runName}"?`);
    if (!confirmed) {
      return;
    }

    setBundleError("");
    setBundleSuccess("");
    setDeletingBundleId(bundle.id);
    try {
      const response = await authFetch(`/api/run-bundles/${bundle.id}`, { method: "DELETE" });
      if (!response.ok) {
        const detail = await safeErrorMessage(response);
        throw new Error(detail);
      }
      setBundleSuccess(`Removed run bundle ${bundle.runName}.`);
      setSelectedBundleId((current) => (current === bundle.id ? null : current));
      await refreshDatasetsAndBundles();
    } catch (error) {
      setBundleError(error.message || "Run bundle removal failed.");
    } finally {
      setDeletingBundleId("");
    }
  }

  if (!user) {
    return (
      <AuthScreen
        authMode={authMode}
        authForm={authForm}
        authError={authError}
        authWorking={authWorking}
        isBootLoading={isBootLoading}
        onModeChange={setAuthMode}
        onFieldChange={(field, value) => setAuthForm((current) => ({ ...current, [field]: value }))}
        onSubmit={handleAuthSubmit}
      />
    );
  }

  return (
    <div className="app-shell">
      <Topbar
        activeScreen={activeScreen}
        onNavigate={setActiveScreen}
        selectedRun={selectedRun}
        apiStatus={apiStatus}
        user={user}
        onLogout={handleLogout}
      />

      <div className="shell">
        <Sidebar
          activeScreen={activeScreen}
          onNavigate={setActiveScreen}
          runningJobs={runningJobs}
          onOpenJobs={() => {
            setActiveScreen("datasets");
            setSelectedJobId(runningJobs[0]?.id ?? selectedJobId);
          }}
        />

        <main className="main">
          <section className={`screen ${activeScreen === "overview" ? "active" : ""}`}>
            <PageHeader
              breadcrumb="Platform / Overview"
              title="Domain Adaptive"
              emphasized="Control Room"
              subtitle="Use DALTP as a stable frontend shell while heavier generation and training work can stay local or move into Colab when hardware is constrained."
            />

            <div className="kpi-strip">
              {dashboard.stats.map((stat) => (
                <KpiCell key={stat.label} stat={stat} />
              ))}
            </div>

            <div className="single-col">
              <Card
                title="Current evaluation health"
                subtitle={selectedRun ? `${selectedRun.name} - benchmark ${summary.benchmarkSize ?? 0} samples` : "No evaluation runs in this account yet"}
                action={
                  selectedRun ? (
                    <button
                      type="button"
                      className="card-action-btn"
                      onClick={() => setActiveScreen("evaluation")}
                    >
                      Compare outputs
                    </button>
                  ) : null
                }
              >
                {selectedRun && (summary.systems ?? []).length ? (
                  (summary.systems ?? []).map((system) => (
                    <EvalBarGroup
                      key={system.name}
                      label={prettifyLabel(system.name)}
                      metrics={system.metrics}
                    />
                  ))
                ) : (
                  <EmptyState text="Run an evaluation from the Evaluation page to see model metrics and compare outputs here." />
                )}
              </Card>
            </div>

            <div className="two-col">
              <Card title="Recent scored runs" subtitle="Only completed evaluation reports from this account are shown here.">
                {dashboard.recentRuns.length ? (
                  <div className="run-list">
                    {dashboard.recentRuns.map((run) => (
                      <button
                        key={run.id}
                        type="button"
                        className={`run-row ${selectedRunId === run.id ? "selected" : ""}`}
                        onClick={() => {
                          setSelectedRunId(run.id);
                          setActiveScreen("evaluation");
                        }}
                      >
                        <div>
                          <div className="run-row-title">{run.name}</div>
                          <div className="run-row-sub">
                            {run.model} - {run.benchmarkSize} samples
                          </div>
                        </div>
                        <div className="run-row-right">
                          <StatusPill status={run.status} />
                          <div className="run-row-metric">
                            {run.metricLabel}: <strong>{run.metric}</strong>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <EmptyState text="No scored runs are attached to this account yet. Launch an evaluation job to populate this view." />
                )}
              </Card>

              <Card title="Prepared launch bundles" subtitle="Reusable local and Colab handoff packages">
                {selectedBundle ? (
                  <BundleSummary
                    bundle={selectedBundle}
                    onDownload={handleDownloadBundle}
                    downloading={downloadingBundleId === selectedBundle.id}
                  />
                ) : (
                  <EmptyState text="No bundle has been prepared yet. Use the run builder to create one." />
                )}
              </Card>
            </div>
          </section>

          <section className={`screen ${activeScreen === "training" ? "active" : ""}`}>
            <PageHeader
              breadcrumb="Platform / Training"
              title="Run Builder"
              emphasized="Studio"
              subtitle="Configure a DALTP run once, then decide whether it should execute on the user's hardware or be exported into a Colab-assisted bundle."
            />

            <div className="two-col training-layout">
              <div>
                <Card title="Build a DALTP run" subtitle="The fields below map directly to your current trainer config format.">
                  <form className="form-grid" onSubmit={handleCreateBundle}>
                    <FormField label="Training trial name">
                      <input
                        className="text-input"
                        value={trainingForm.runName}
                        onChange={(event) => updateTrainingField("runName", event.target.value)}
                        placeholder="endorsement-risk-trial"
                      />
                    </FormField>

                    <FormField label="Execution mode">
                      <div className="choice-grid three">
                        {options.executionModes.map((mode) => (
                          <button
                            key={mode.id}
                            type="button"
                            className={`choice-card ${trainingForm.executionMode === mode.id ? "selected" : ""}`}
                            onClick={() => updateTrainingField("executionMode", mode.id)}
                          >
                            <strong>{mode.label}</strong>
                            <span>{mode.description}</span>
                          </button>
                        ))}
                      </div>
                    </FormField>

                    <FormField label="Base model">
                      <div className="locked-model-card">
                        <strong>{primaryModel.name}</strong>
                        <span>
                          {primaryModel.provider} - {primaryModel.params}
                        </span>
                        <span>{primaryModel.vramHint}</span>
                        <em>DALTP is currently tuned around this training pipeline.</em>
                      </div>
                    </FormField>

                    <div className="inline-form-row">
                      <FormField label="PEFT method">
                        <div className="pill-row">
                          {options.peftMethods.map((method) => (
                            <button
                              key={method.id}
                              type="button"
                              className={`pill-button ${trainingForm.peftMethod === method.id ? "selected" : ""}`}
                              onClick={() => updateTrainingField("peftMethod", method.id)}
                            >
                              {method.label}
                            </button>
                          ))}
                        </div>
                      </FormField>

                      <FormField label="LoRA rank">
                        <div className="pill-row">
                          {options.loraRanks.map((rank) => (
                            <button
                              key={rank}
                              type="button"
                              className={`pill-button ${Number(trainingForm.loraRank) === rank ? "selected" : ""}`}
                              onClick={() => updateTrainingField("loraRank", rank)}
                            >
                              {rank}
                            </button>
                          ))}
                        </div>
                      </FormField>
                    </div>

                    <div className="inline-form-row">
                      <FormField label="QA dataset">
                        {qaDatasets.length ? (
                          <CustomSelect
                            value={trainingForm.qaDatasetId}
                            options={qaDatasets.map((dataset) => ({ value: dataset.id, label: dataset.name }))}
                            onChange={(value) => updateTrainingField("qaDatasetId", value)}
                            placeholder="Choose a QA dataset"
                          />
                        ) : (
                          <div className="field-empty-note">Upload a QA dataset first from the Datasets page.</div>
                        )}
                      </FormField>

                      <FormField label="Instruction dataset">
                        {instructionDatasets.length ? (
                          <CustomSelect
                            value={trainingForm.instructionDatasetId}
                            options={instructionDatasets.map((dataset) => ({ value: dataset.id, label: dataset.name }))}
                            onChange={(value) => updateTrainingField("instructionDatasetId", value)}
                            placeholder="Choose an instruction dataset"
                          />
                        ) : (
                          <div className="field-empty-note">Upload an instruction dataset first from the Datasets page.</div>
                        )}
                      </FormField>
                    </div>

                    <FormField label="Config mode">
                      <div className="pill-row">
                        {[
                          { id: "preset", label: "Preset" },
                          { id: "manual", label: "Manual" },
                          { id: "upload", label: "Upload JSON" },
                        ].map((mode) => (
                          <button
                            key={mode.id}
                            type="button"
                            className={`pill-button ${trainingForm.configMode === mode.id ? "selected" : ""}`}
                            onClick={() => updateTrainingField("configMode", mode.id)}
                          >
                            {mode.label}
                          </button>
                        ))}
                      </div>
                    </FormField>

                    {trainingForm.configMode === "preset" ? (
                      <FormField label="Preset profile">
                        <div className="choice-grid two">
                          {options.configPresets.map((preset) => (
                            <button
                              key={preset.id}
                              type="button"
                              className={`choice-card ${trainingForm.presetId === preset.id ? "selected" : ""}`}
                              onClick={() => handlePresetChange(preset.id)}
                            >
                              <strong>{preset.label}</strong>
                              <span>{preset.description}</span>
                            </button>
                          ))}
                        </div>
                      </FormField>
                    ) : null}

                    {trainingForm.configMode === "manual" ? (
                      <div className="manual-config-grid">
                        <FormField label="Learning rate">
                          <input
                            className="text-input"
                            type="number"
                            step="0.00001"
                            value={trainingForm.manualConfig.learningRate}
                            onChange={(event) =>
                              updateManualConfig("learningRate", Number(event.target.value))
                            }
                          />
                        </FormField>
                        <FormField label="Epochs">
                          <input
                            className="text-input"
                            type="number"
                            value={trainingForm.manualConfig.epochs}
                            onChange={(event) => updateManualConfig("epochs", Number(event.target.value))}
                          />
                        </FormField>
                        <FormField label="Batch size">
                          <input
                            className="text-input"
                            type="number"
                            value={trainingForm.manualConfig.batchSize}
                            onChange={(event) => updateManualConfig("batchSize", Number(event.target.value))}
                          />
                        </FormField>
                        <FormField label="Grad accumulation">
                          <input
                            className="text-input"
                            type="number"
                            value={trainingForm.manualConfig.gradientAccumulationSteps}
                            onChange={(event) =>
                              updateManualConfig("gradientAccumulationSteps", Number(event.target.value))
                            }
                          />
                        </FormField>
                        <FormField label="Max length">
                          <input
                            className="text-input"
                            type="number"
                            value={trainingForm.manualConfig.maxLength}
                            onChange={(event) => updateManualConfig("maxLength", Number(event.target.value))}
                          />
                        </FormField>
                        <FormField label="Dtype">
                          <CustomSelect
                            value={trainingForm.manualConfig.dtype}
                            options={[
                              { value: "bfloat16", label: "bfloat16" },
                              { value: "float16", label: "float16" },
                            ]}
                            onChange={(value) => updateManualConfig("dtype", value)}
                          />
                        </FormField>
                        <FormField label="4-bit quantization">
                          <div className="pill-row">
                            {[true, false].map((value) => (
                              <button
                                key={String(value)}
                                type="button"
                                className={`pill-button ${trainingForm.manualConfig.loadIn4bit === value ? "selected" : ""}`}
                                onClick={() => updateManualConfig("loadIn4bit", value)}
                              >
                                {value ? "Enabled" : "Disabled"}
                              </button>
                            ))}
                          </div>
                        </FormField>
                      </div>
                    ) : null}

                    {trainingForm.configMode === "upload" ? (
                      <FormField label="Uploaded config JSON">
                        <textarea
                          className="text-area"
                          rows={10}
                          value={trainingForm.uploadedConfigText}
                          onChange={(event) => updateTrainingField("uploadedConfigText", event.target.value)}
                          placeholder='{"model": {...}, "datasets": {...}, "training": {...}}'
                        />
                      </FormField>
                    ) : null}

                    {bundleError ? <div className="form-message error">{bundleError}</div> : null}
                    {bundleSuccess ? <div className="form-message success">{bundleSuccess}</div> : null}

                    <div className="form-actions">
                      <button
                        type="button"
                        className="btn btn-outline"
                        onClick={() => setActiveScreen("datasets")}
                      >
                        Upload datasets
                      </button>
                      <button type="submit" className="btn btn-primary" disabled={creatingBundle || !canPrepareRun}>
                        {creatingBundle ? "Preparing bundle..." : "Prepare run bundle"}
                      </button>
                    </div>
                  </form>
                </Card>
              </div>

              <div>
                <Card title="Prepared bundles" subtitle="Download, copy commands, or hand off to Colab from here.">
                  {bundles.length ? (
                    <div className="bundle-list">
                      {bundles.map((bundle) => (
                        <button
                          key={bundle.id}
                          type="button"
                          className={`bundle-item ${selectedBundle?.id === bundle.id ? "selected" : ""}`}
                          onClick={() => setSelectedBundleId(bundle.id)}
                        >
                          <div className="bundle-item-main">
                            <strong>{bundle.runName}</strong>
                            <span>{prettifyLabel(bundle.executionMode)} - {bundle.baseModel.split("/").pop()}</span>
                          </div>
                          <StatusPill status="Done" />
                        </button>
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="Prepare a bundle and it will appear here with download and launch instructions." />
                  )}

                  {selectedBundle ? (
                    <div className="bundle-detail">
                      <div className="section-rule compact">
                        <div className="section-rule-title">Selected bundle</div>
                        <div className="section-rule-line" />
                      </div>

                      <dl className="detail-list">
                        <div>
                          <dt>Execution</dt>
                          <dd>{prettifyLabel(selectedBundle.executionMode)}</dd>
                        </div>
                        <div>
                          <dt>PEFT</dt>
                          <dd>{selectedBundle.peftMethod.toUpperCase()}</dd>
                        </div>
                        <div>
                          <dt>LoRA rank</dt>
                          <dd>{selectedBundle.loraRank}</dd>
                        </div>
                      </dl>

                      <div className="bundle-actions">
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={() => handleDownloadBundle(selectedBundle)}
                          disabled={downloadingBundleId === selectedBundle.id}
                        >
                          {downloadingBundleId === selectedBundle.id ? "Downloading..." : "Download zip"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger"
                          onClick={() => handleDeleteBundle(selectedBundle)}
                          disabled={deletingBundleId === selectedBundle.id}
                        >
                          {deletingBundleId === selectedBundle.id ? "Removing..." : "Remove bundle"}
                        </button>
                        {selectedBundle.executionMode === "local" ? (
                          <button
                            type="button"
                            className="btn btn-outline"
                            onClick={() => handleLaunchLocalTraining(selectedBundle)}
                            disabled={launchingBundleId === selectedBundle.id}
                          >
                            {launchingBundleId === selectedBundle.id ? "Launching..." : "Start local training"}
                          </button>
                        ) : null}
                      </div>

                      <CommandPanel
                        title="Local command set"
                        commands={selectedBundle.commands?.local ?? []}
                      />
                      <CommandPanel
                        title="Colab command set"
                        commands={selectedBundle.commands?.colab ?? []}
                      />
                    </div>
                  ) : null}
                </Card>

              </div>
            </div>
          </section>

          <section className={`screen ${activeScreen === "models" ? "active" : ""}`}>
            <PageHeader
              breadcrumb="Platform / Models"
              title="Model"
              emphasized="Registry"
              subtitle="Track trained adapters, download local artifacts, and import successful Colab runs back into DALTP."
            />

            <div className="two-col training-layout">
              <div>
                <Card
                  title="Import trained model"
                  subtitle="Local training registers automatically. Use this form to bring completed Colab artifacts back into the account."
                >
                  <form className="form-grid" onSubmit={handleImportModelArtifact}>
                    <FormField label="Model artifact name">
                      <input
                        className="text-input"
                        value={modelImportForm.name}
                        onChange={(event) => setModelImportForm((current) => ({ ...current, name: event.target.value }))}
                        placeholder="endorsement-qlora-v1"
                      />
                    </FormField>

                    <div className="inline-form-row">
                      <FormField label="Source">
                        <div className="choice-grid two">
                          {[
                            {
                              id: "colab",
                              label: "Colab-assisted",
                              description: "Import artifact files after training finishes in Colab.",
                            },
                            {
                              id: "manual",
                              label: "Manual import",
                              description: "Register adapter files from another local or external run.",
                            },
                          ].map((source) => (
                            <button
                              key={source.id}
                              type="button"
                              className={`choice-card ${modelImportForm.source === source.id ? "selected" : ""}`}
                              onClick={() => setModelImportForm((current) => ({ ...current, source: source.id }))}
                            >
                              <strong>{source.label}</strong>
                              <span>{source.description}</span>
                            </button>
                          ))}
                        </div>
                      </FormField>

                      <FormField label="Base model">
                        <div className="locked-model-card">
                          <strong>{primaryModel.name}</strong>
                          <span>{primaryModel.provider} - {primaryModel.params}</span>
                          <span>{primaryModel.vramHint}</span>
                        </div>
                      </FormField>
                    </div>

                    <div className="field-empty-note">
                      {modelImportForm.source === "colab"
                        ? "Use this when DALTP prepared the run bundle, training happened in Colab, and you now want to bring the finished adapter files back into your account."
                        : "Use this when you already have trained adapter files outside DALTP and want to register them for download and reuse."}
                    </div>

                    <div className="inline-form-row">
                      <FormField label="PEFT method">
                        <div className="pill-row">
                          {options.peftMethods.map((method) => (
                            <button
                              key={method.id}
                              type="button"
                              className={`pill-button ${modelImportForm.peftMethod === method.id ? "selected" : ""}`}
                              onClick={() => setModelImportForm((current) => ({ ...current, peftMethod: method.id }))}
                            >
                              {method.label}
                            </button>
                          ))}
                        </div>
                      </FormField>

                      <FormField label="LoRA rank">
                        <div className="pill-row">
                          {options.loraRanks.map((rank) => (
                            <button
                              key={rank}
                              type="button"
                              className={`pill-button ${Number(modelImportForm.loraRank) === rank ? "selected" : ""}`}
                              onClick={() => setModelImportForm((current) => ({ ...current, loraRank: rank }))}
                            >
                              {rank}
                            </button>
                          ))}
                        </div>
                      </FormField>
                    </div>

                    <FormField label="Artifact files">
                      <div className="upload-dropzone">
                        <strong>Choose trained model files</strong>
                        <span>
                          Upload the adapter files produced by training, or a single exported zip from Colab. DALTP can unpack a model zip during import.
                        </span>
                        <em>{modelImportSummaryText}</em>
                        <div className="upload-actions-inline">
                          <button type="button" className="btn btn-outline btn-tight" onClick={() => document.getElementById("model-files")?.click()}>
                            Choose model zip or files
                          </button>
                          <button type="button" className="upload-link-btn" onClick={() => document.getElementById("model-folder")?.click()}>
                            Choose a folder instead
                          </button>
                        </div>
                      </div>
                      <input
                        id="model-files"
                        className="file-input hidden-input"
                        type="file"
                        multiple
                        onChange={(event) => handleModelImportFileSelection(event.target.files)}
                      />
                      <input
                        id="model-folder"
                        className="file-input hidden-input"
                        type="file"
                        multiple
                        webkitdirectory=""
                        directory=""
                        onChange={(event) => handleModelImportFileSelection(event.target.files)}
                      />
                      <div className="file-list">
                        {modelImportForm.files.length ? (
                          modelImportForm.files.map((entry) => (
                            <div key={entry.relativePath} className="file-item">
                              <span className="file-name">{entry.displayName}</span>
                              <strong>{formatBytes(entry.file.size)}</strong>
                            </div>
                          ))
                        ) : (
                          <span className="file-empty">Choose a model zip, the adapter files, or a folder exported from Colab/local training.</span>
                        )}
                      </div>
                    </FormField>

                    {modelError ? <div className="form-message error">{modelError}</div> : null}
                    {bundleSuccess && !bundleError && !uploadError ? <div className="form-message success">{bundleSuccess}</div> : null}

                    <div className="form-actions">
                      <button type="submit" className="btn btn-primary" disabled={importingModel}>
                        {importingModel ? "Importing..." : "Import model"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline"
                        onClick={() =>
                          setModelImportForm({
                            name: "",
                            source: "colab",
                            baseModel: options.models?.[0]?.id ?? initialOptions.models[0].id,
                            peftMethod: "qlora",
                            loraRank: 16,
                            files: [],
                          })
                        }
                      >
                        Clear form
                      </button>
                    </div>
                  </form>
                </Card>
              </div>

              <div>
                <Card title="Trained model registry" subtitle={`${models.length} model artifacts available in this account`}>
                  {models.length ? (
                    <div className="bundle-list">
                      {models.map((model) => (
                        <button
                          key={model.id}
                          type="button"
                          className={`bundle-item ${selectedModel?.id === model.id ? "selected" : ""}`}
                          onClick={() => setSelectedModelId(model.id)}
                        >
                          <div className="bundle-item-main">
                            <strong>{model.name}</strong>
                            <span>{prettifyLabel(model.source)} - {prettifyLabel(model.peftMethod)} - rank {model.loraRank ?? "--"}</span>
                          </div>
                          <StatusPill status="Done" />
                        </button>
                      ))}
                    </div>
                  ) : (
                    <EmptyState text="No trained model artifacts are attached to this account yet. Run training locally or import a completed Colab artifact." />
                  )}

                  {selectedModel ? (
                    <div className="bundle-detail">
                      <div className="section-rule compact">
                        <div className="section-rule-title">Selected model artifact</div>
                        <div className="section-rule-line" />
                      </div>

                      <dl className="detail-list">
                        <div>
                          <dt>Base model</dt>
                          <dd>{selectedModel.baseModel?.split("/").pop() ?? "--"}</dd>
                        </div>
                        <div>
                          <dt>Source</dt>
                          <dd>{prettifyLabel(selectedModel.source)}</dd>
                        </div>
                        <div>
                          <dt>PEFT</dt>
                          <dd>{selectedModel.peftMethod?.toUpperCase() ?? "--"}</dd>
                        </div>
                        <div>
                          <dt>LoRA rank</dt>
                          <dd>{selectedModel.loraRank ?? "--"}</dd>
                        </div>
                        <div>
                          <dt>Storage</dt>
                          <dd>{selectedModel.storageBucket ?? "--"}</dd>
                        </div>
                        <div>
                          <dt>Files</dt>
                          <dd>{selectedModel.fileCount ?? 0}</dd>
                        </div>
                        <div>
                          <dt>Archive size</dt>
                          <dd>{selectedModel.archiveSizeMb ?? 0} MB</dd>
                        </div>
                      </dl>

                      <div className="bundle-actions">
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={() => handleDownloadModel(selectedModel)}
                          disabled={downloadingModelId === selectedModel.id}
                        >
                          {downloadingModelId === selectedModel.id ? "Downloading..." : "Download model zip"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger"
                          onClick={() => handleDeleteModel(selectedModel)}
                          disabled={deletingModelId === selectedModel.id}
                        >
                          {deletingModelId === selectedModel.id ? "Removing..." : "Remove model"}
                        </button>
                      </div>

                      {selectedModel.source === "colab" ? (
                        <div className="field-empty-note">
                          This artifact was imported back into DALTP after a Colab-assisted run. DALTP keeps Colab as a handoff path, then registers the finished files here for download and reuse.
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </Card>
              </div>
            </div>
          </section>

          <section className={`screen ${activeScreen === "evaluation" ? "active" : ""}`}>
            <PageHeader
              breadcrumb="Platform / Evaluation"
              title="Evaluation"
              emphasized="Reports"
              subtitle="Review the benchmark summary, compare systems, and inspect actual answers against model outputs."
            />

            <div className="two-col">
              <Card title="Launch evaluation" subtitle="Run predictions and scoring against a benchmark dataset already stored in DALTP.">
                <form className="form-grid" onSubmit={handleCreateEvaluationJob}>
                  <FormField label="Evaluation trial name">
                    <input
                      className="text-input"
                      value={evaluationForm.runName}
                      onChange={(event) => setEvaluationForm((current) => ({ ...current, runName: event.target.value }))}
                      placeholder="endorsement-rag-eval"
                    />
                  </FormField>

                  <FormField label="Benchmark dataset">
                    {benchmarkDatasets.length ? (
                      <CustomSelect
                        value={evaluationForm.benchmarkDatasetId}
                        options={benchmarkDatasets.map((dataset) => ({ value: dataset.id, label: dataset.name }))}
                        onChange={(value) => setEvaluationForm((current) => ({ ...current, benchmarkMode: "existing", benchmarkDatasetId: value }))}
                        placeholder="Choose a benchmark dataset"
                      />
                    ) : (
                      <div className="field-empty-note">No benchmark dataset is in this account yet. Generate or upload one from Dataset Registry first.</div>
                    )}
                  </FormField>

                  <FormField label="Corpus / RAG dataset">
                    {corpusDatasets.length ? (
                      <CustomSelect
                        value={evaluationForm.corpusDatasetId}
                        options={corpusDatasets.map((dataset) => ({ value: dataset.id, label: dataset.name }))}
                        onChange={(value) => setEvaluationForm((current) => ({ ...current, corpusDatasetId: value }))}
                        placeholder="Choose a corpus dataset"
                      />
                    ) : (
                      <div className="field-empty-note">Generate or upload a corpus dataset first.</div>
                    )}
                  </FormField>

                  {evaluationForm.runFineTuned || evaluationForm.runFineTunedRag ? (
                    <FormField label="Model artifact for fine-tuned modes">
                      {deployableEvaluationModels.length ? (
                        <CustomSelect
                          value={evaluationForm.modelId}
                          options={deployableEvaluationModels.map((model) => ({
                            value: model.id,
                            label: `${model.name} (${prettifyLabel(model.peftMethod)} - rank ${model.loraRank ?? "--"})`,
                          }))}
                          onChange={(value) => setEvaluationForm((current) => ({ ...current, modelId: value }))}
                          placeholder="Choose a model artifact"
                        />
                      ) : (
                        <div className="field-empty-note">Import a trained model into Model Registry before running fine-tuned evaluation.</div>
                      )}
                    </FormField>
                  ) : null}

                  <FormField label="Systems to score">
                    <div className="choice-grid two">
                      {[
                        ["runBase", "Base"],
                        ["runRag", "RAG"],
                        ["runFineTuned", "Fine-tuned"],
                        ["runFineTunedRag", "Fine-tuned + RAG"],
                      ].map(([field, label]) => (
                        <button
                          key={field}
                          type="button"
                          className={`choice-card ${evaluationForm[field] ? "selected" : ""}`}
                          onClick={() => {
                            setEvaluationForm((current) => ({ ...current, [field]: !current[field] }));
                            setEvaluationNotice("");
                          }}
                        >
                          <strong>{label}</strong>
                          <span>{evaluationForm[field] ? "Included in this run" : "Excluded from this run"}</span>
                        </button>
                      ))}
                    </div>
                    <div className="field-empty-note">
                      Base and RAG predictions run through OpenRouter. Fine-tuned modes use your configured Modal evaluation endpoint.
                    </div>
                  </FormField>

                  {evaluationNotice ? <div className="form-message success">{evaluationNotice}</div> : null}
                  {evaluationError ? <div className="form-message error">{evaluationError}</div> : null}

                  <div className="form-actions">
                    <button type="submit" className="btn btn-primary" disabled={creatingEvaluationJob}>
                      {creatingEvaluationJob ? "Starting evaluation..." : "Run evaluation"}
                    </button>
                  </div>
                </form>
              </Card>

              <Card title="Evaluation jobs" subtitle="Only evaluation runs are shown here so scoring activity is easier to follow.">
                <JobPanel
                  jobs={evaluationJobs}
                  selectedJob={evaluationJobs.find((job) => job.id === selectedJob?.id) ?? evaluationJobs[0] ?? null}
                  onSelectJob={setSelectedJobId}
                  emptyText="No evaluation jobs yet. Launch a benchmark run from this screen to populate scoring activity."
                />
              </Card>
            </div>

            <Card
              title="System comparison"
              subtitle={summary.runName ? `${summary.runName} - ${summary.benchmarkSize} benchmark samples` : "No evaluation results available yet"}
              action={
                summary.runId ? (
                  <div className="bundle-actions">
                    <button
                      type="button"
                      className="btn btn-outline"
                      onClick={() => handleDownloadEvaluationRun(summary.runId, summary.runName)}
                      disabled={downloadingRunId === summary.runId}
                    >
                      {downloadingRunId === summary.runId ? "Downloading..." : "Download report"}
                    </button>
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={() => handleDeleteEvaluationRun(summary.runId, summary.runName)}
                      disabled={deletingRunId === summary.runId}
                    >
                      {deletingRunId === summary.runId ? "Removing..." : "Remove run"}
                    </button>
                  </div>
                ) : null
              }
            >
                {(summary.systems ?? []).length ? (
                  <div className="compare-table-wrap">
                    <table className="compare-table">
                      <thead>
                        <tr>
                          <th>System</th>
                          <th>ROUGE-L</th>
                          <th>BERTScore F1</th>
                          <th>Fact Coverage</th>
                          <th>Samples</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(summary.systems ?? []).map((system) => (
                          <tr key={system.name}>
                            <td><span className="metric-name">{prettifyLabel(system.name)}</span></td>
                            <td className="metric-val-ok">{formatMetric(system.metrics?.["ROUGE-L"])}</td>
                            <td className="metric-val-good">{formatMetric(system.metrics?.["BERTScore F1"])}</td>
                            <td className="metric-val-ok">{formatMetric(system.metrics?.["Fact Coverage"])}</td>
                            <td>{system.metrics?.samples_scored ?? summary.benchmarkSize}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <EmptyState text="Launch an evaluation job to populate per-system scoring here." />
                )}
            </Card>
          </section>

          <section className={`screen ${activeScreen === "datasets" ? "active" : ""}`}>
            <PageHeader
              breadcrumb="Platform / Datasets"
              title="Dataset"
              emphasized="Registry"
              subtitle="Mix DALTP-generated datasets with user-provided corpora, then plug them directly into the run builder."
            />

            {runningDatasetJobs.length ? (
              <div className="callout-box compact">
                <strong>{runningDatasetJobs.length === 1 ? runningDatasetJobs[0].title : `${runningDatasetJobs.length} dataset jobs are running`}</strong>
                <span>DALTP is still generating or ingesting data in the background. You can stay on this screen and the registry will refresh automatically.</span>
              </div>
            ) : null}

            <div className="datasets-layout">
              <Card
                title={datasetFormMode === "upload" ? "Upload a dataset" : "Generate from documents"}
                subtitle={
                  datasetFormMode === "upload"
                    ? "Add a ready-made QA, instruction, or benchmark dataset to the registry."
                    : "Create corpus, QA, instruction, or benchmark datasets from raw source files inside DALTP."
                }
              >
                <div className="dataset-form-switch">
                  <button
                    type="button"
                    className={`pill-button ${datasetFormMode === "upload" ? "selected" : ""}`}
                    onClick={() => setDatasetFormMode("upload")}
                  >
                    Upload dataset
                  </button>
                  <button
                    type="button"
                    className={`pill-button ${datasetFormMode === "generate" ? "selected" : ""}`}
                    onClick={() => setDatasetFormMode("generate")}
                  >
                    Generate from documents
                  </button>
                </div>

                {datasetFormMode === "upload" ? (
                  <form className="form-grid" onSubmit={handleDatasetUpload}>
                  <FormField label="Dataset name">
                    <input
                      className="text-input"
                      value={datasetUpload.name}
                      onChange={(event) => setDatasetUpload((current) => ({ ...current, name: event.target.value }))}
                      placeholder="custom-endorsement-qa"
                    />
                  </FormField>

                  <FormField label="Dataset kind">
                    <CustomSelect
                      value={datasetUpload.kind}
                      options={[
                        { value: "qa", label: "QA dataset" },
                        { value: "instruction", label: "Instruction dataset" },
                        { value: "benchmark", label: "Benchmark dataset" },
                      ]}
                      onChange={(value) => setDatasetUpload((current) => ({ ...current, kind: value }))}
                    />
                  </FormField>

                  <FormField label="Files">
                    <div className="upload-dropzone">
                      <strong>Choose dataset files</strong>
                      <span>
                        Upload `.json`, `.jsonl`, `.csv`, or `.txt` content for DALTP to package into a run. Folder uploads keep their relative paths and are merged into a single dataset file behind the scenes.
                      </span>
                      <em>{uploadSummaryText}</em>
                      <div className="upload-actions-inline">
                        <button type="button" className="btn btn-outline btn-tight" onClick={() => document.getElementById("dataset-files")?.click()}>
                          Choose dataset files
                        </button>
                        <button type="button" className="upload-link-btn" onClick={() => document.getElementById("dataset-folder")?.click()}>
                          Choose a folder instead
                        </button>
                      </div>
                    </div>
                    <input
                      id="dataset-files"
                      className="file-input hidden-input"
                      type="file"
                      multiple
                      accept=".json,.jsonl,.csv,.txt"
                      onChange={(event) => handleDatasetFileSelection(event.target.files)}
                    />
                    <input
                      id="dataset-folder"
                      className="file-input hidden-input"
                      type="file"
                      multiple
                      webkitdirectory=""
                      directory=""
                      accept=".json,.jsonl,.csv,.txt"
                      onChange={(event) => handleDatasetFileSelection(event.target.files)}
                    />
                    <div className="file-list">
                      {datasetUpload.files.length ? (
                        datasetUpload.files.map((entry) => (
                          <div key={entry.relativePath} className="file-item">
                            <span className="file-name">{entry.displayName}</span>
                            <strong>{formatBytes(entry.file.size)}</strong>
                            <button
                              type="button"
                              className="btn btn-outline btn-tight"
                              onClick={() => removeDatasetUploadFile(entry.relativePath)}
                            >
                              Remove
                            </button>
                          </div>
                        ))
                      ) : (
                        <span className="file-empty">Choose one or more files, or bring in a dataset folder from the same upload area.</span>
                      )}
                    </div>
                  </FormField>

                  {uploadError ? <div className="form-message error">{uploadError}</div> : null}
                  {bundleSuccess && !bundleError ? <div className="form-message success">{bundleSuccess}</div> : null}

                  <div className="form-actions">
                    <button type="submit" className="btn btn-primary" disabled={uploadingDataset}>
                      {uploadingDataset ? "Uploading..." : "Upload dataset"}
                    </button>
                    <button
                      type="button"
                      className="btn btn-outline"
                      onClick={() => setDatasetUpload({ name: "", kind: "qa", files: [] })}
                    >
                      Clear form
                    </button>
                    <button type="button" className="btn btn-outline" onClick={() => setActiveScreen("training")}>
                      Use in run builder
                    </button>
                  </div>
                  </form>
                ) : (
                  <form className="form-grid" onSubmit={handleGenerateDataset}>
                    <FormField label="Generated dataset name">
                      <input
                        className="text-input"
                        value={datasetGeneration.name}
                        onChange={(event) => setDatasetGeneration((current) => ({ ...current, name: event.target.value }))}
                          placeholder="consulting-agreement-qa"
                        />
                      </FormField>

                    <FormField label="Dataset kind to generate">
                      <CustomSelect
                        value={datasetGeneration.kind}
                        options={[
                          { value: "qa", label: "QA dataset" },
                          { value: "instruction", label: "Instruction dataset" },
                          { value: "benchmark", label: "Benchmark dataset" },
                          { value: "corpus", label: "Corpus dataset" },
                        ]}
                        onChange={(value) => setDatasetGeneration((current) => ({ ...current, kind: value }))}
                      />
                    </FormField>

                    {datasetGeneration.kind === "benchmark" ? (
                      <FormField label="Corpus dataset for benchmark generation">
                        {corpusDatasets.length ? (
                          <CustomSelect
                            value={datasetGeneration.corpusDatasetId}
                            options={corpusDatasets.map((dataset) => ({ value: dataset.id, label: dataset.name }))}
                            onChange={(value) => setDatasetGeneration((current) => ({ ...current, corpusDatasetId: value }))}
                            placeholder="Choose a corpus dataset"
                          />
                        ) : (
                          <div className="field-empty-note">Create or upload a corpus dataset first, then choose it here for benchmark generation.</div>
                        )}
                      </FormField>
                    ) : null}

                    {datasetGeneration.kind !== "benchmark" ? (
                    <FormField label="Source documents">
                      <div className="upload-dropzone">
                        <strong>Choose source documents</strong>
                        <span>
                          Use PDF, DOCX, TXT, CSV, or XLSX source files and let DALTP turn them into a trainer-ready dataset.
                        </span>
                        <em>{generationSummaryText}</em>
                        <div className="upload-actions-inline">
                          <button type="button" className="btn btn-outline btn-tight" onClick={() => document.getElementById("generate-files")?.click()}>
                            Choose source files
                          </button>
                          <button type="button" className="upload-link-btn" onClick={() => document.getElementById("generate-folder")?.click()}>
                            Choose a folder instead
                          </button>
                        </div>
                      </div>
                      <input
                        id="generate-files"
                        className="file-input hidden-input"
                        type="file"
                        multiple
                        accept=".pdf,.docx,.txt,.csv,.xlsx,.json,.jsonl"
                        onChange={(event) => handleGenerationFileSelection(event.target.files)}
                      />
                      <input
                        id="generate-folder"
                        className="file-input hidden-input"
                        type="file"
                        multiple
                        webkitdirectory=""
                        directory=""
                        accept=".pdf,.docx,.txt,.csv,.xlsx,.json,.jsonl"
                        onChange={(event) => handleGenerationFileSelection(event.target.files)}
                      />
                      <div className="file-list">
                        {datasetGeneration.files.length ? (
                          datasetGeneration.files.map((entry) => (
                            <div key={entry.relativePath} className="file-item">
                              <span className="file-name">{entry.displayName}</span>
                              <strong>{formatBytes(entry.file.size)}</strong>
                              <button
                                type="button"
                                className="btn btn-outline btn-tight"
                                onClick={() => removeGenerationFile(entry.relativePath)}
                              >
                                Remove
                              </button>
                            </div>
                          ))
                        ) : (
                          <span className="file-empty">Choose source files, or bring in a whole document folder from the same upload area.</span>
                        )}
                      </div>
                    </FormField>
                    ) : null}

                    {datasetGeneration.kind === "qa" || datasetGeneration.kind === "benchmark" ? (
                      <>
                        <div className="inline-form-row">
                          <FormField label={datasetGeneration.kind === "benchmark" ? "Benchmark pairs per context" : "QA pairs per context"}>
                            <input
                              className="text-input"
                              type="number"
                              min="1"
                              value={datasetGeneration.qaNumPairs}
                              onChange={(event) => setDatasetGeneration((current) => ({ ...current, qaNumPairs: Number(event.target.value) }))}
                            />
                          </FormField>
                          <FormField label="QA chunk size">
                            <input
                              className="text-input"
                              type="number"
                              min="500"
                              value={datasetGeneration.qaChunkSize}
                              onChange={(event) => setDatasetGeneration((current) => ({ ...current, qaChunkSize: Number(event.target.value) }))}
                            />
                          </FormField>
                        </div>
                        <div className="field-empty-note">
                          {datasetGeneration.kind === "benchmark"
                            ? "Use held-out documents here so the benchmark stays separate from your QA and instruction training datasets."
                            : "Use training documents here to generate QA data for fine-tuning."}
                        </div>
                      </>
                    ) : null}

                    {datasetGeneration.kind === "corpus" ? (
                      <>
                        <div className="inline-form-row">
                          <FormField label="Corpus chunk size">
                            <input
                              className="text-input"
                              type="number"
                              min="100"
                              value={datasetGeneration.corpusChunkSize}
                              onChange={(event) => setDatasetGeneration((current) => ({ ...current, corpusChunkSize: Number(event.target.value) }))}
                            />
                          </FormField>
                          <FormField label="Corpus overlap">
                            <input
                              className="text-input"
                              type="number"
                              min="0"
                              value={datasetGeneration.corpusChunkOverlap}
                              onChange={(event) => setDatasetGeneration((current) => ({ ...current, corpusChunkOverlap: Number(event.target.value) }))}
                            />
                          </FormField>
                        </div>

                        <div className="inline-form-row">
                          <FormField label="RAG storage">
                            <div className="pill-row">
                              <button
                                type="button"
                                className={`pill-button ${datasetGeneration.ingestToPgvector ? "selected" : ""}`}
                                onClick={() => setDatasetGeneration((current) => ({ ...current, ingestToPgvector: true }))}
                              >
                                Ingest to pgvector
                              </button>
                              <button
                                type="button"
                                className={`pill-button ${!datasetGeneration.ingestToPgvector ? "selected" : ""}`}
                                onClick={() => setDatasetGeneration((current) => ({ ...current, ingestToPgvector: false }))}
                              >
                                Keep as corpus only
                              </button>
                            </div>
                          </FormField>
                          <FormField label="Collection name">
                            <input
                              className="text-input"
                              value={datasetGeneration.collectionName}
                              onChange={(event) =>
                                setDatasetGeneration((current) => ({ ...current, collectionName: event.target.value }))
                              }
                              placeholder={slugifyValue(datasetGeneration.name || "rag-corpus")}
                              disabled={!datasetGeneration.ingestToPgvector}
                            />
                          </FormField>
                        </div>

                        <div className="field-empty-note">
                          Corpus datasets power DALTP RAG. If you ingest to pgvector here, the registry will keep the linked
                          retrieval namespace ready for later evaluation runs.
                        </div>
                      </>
                    ) : null}

                    {generationError ? <div className="form-message error">{generationError}</div> : null}

                    <div className="form-actions">
                      <button type="submit" className="btn btn-primary" disabled={generatingDataset}>
                        {generatingDataset ? "Generating..." : "Generate dataset"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline"
                        onClick={() =>
                          setDatasetGeneration({
                            name: "",
                            kind: "qa",
                            files: [],
                            corpusDatasetId: "",
                            modelName: "openrouter/auto",
                            apiBase: "https://openrouter.ai/api/v1",
                            qaNumPairs: 4,
                            qaChunkSize: 2500,
                            qaChunkOverlap: 150,
                            corpusChunkSize: 1000,
                            corpusChunkOverlap: 150,
                            ingestToPgvector: true,
                            collectionName: "",
                          })
                        }
                      >
                        Clear form
                      </button>
                    </div>
                  </form>
                )}
              </Card>

              <Card title="Dataset registry" subtitle={`${datasets.length} datasets available in this account`}>
                <div className="dataset-grid">
                  {datasets.length ? (
                    datasets.map((dataset) => (
                      <div key={dataset.id} className="dataset-card">
                        <div className="ds-icon">{dataset.kind.slice(0, 2).toUpperCase()}</div>
                        <div className="ds-main">
                          <div className="ds-info">
                            <div className="ds-name">{dataset.name}</div>
                            <div className="ds-meta">{dataset.description}</div>
                          </div>
                          <div className="ds-stats">
                            <span className="config-tag">{prettifyLabel(dataset.kind)}</span>
                            <span className="config-tag">
                              {dataset.source === "generated" ? "Generated" : dataset.source === "uploaded" ? "Uploaded" : "DALTP"}
                            </span>
                            <span className="config-tag">{dataset.lineCount ?? 0} rows</span>
                            <span className="config-tag">{dataset.sizeMb ?? 0} MB</span>
                            {dataset.vectorStore?.collectionName ? (
                              <>
                                <span className="config-tag">{dataset.vectorStore?.ingested === false ? "pgvector failed" : "pgvector ready"}</span>
                                <span className="config-tag">{dataset.vectorStore.collectionName}</span>
                              </>
                            ) : null}
                          </div>
                          {dataset.vectorStore?.error ? <div className="ds-meta">pgvector issue: {dataset.vectorStore.error}</div> : null}
                        </div>
                        <div className="dataset-card-actions">
                          <button
                            type="button"
                            className="btn btn-outline"
                            onClick={() => {
                              if (dataset.kind === "qa") {
                                updateTrainingField("qaDatasetId", dataset.id);
                              } else if (dataset.kind === "instruction") {
                                updateTrainingField("instructionDatasetId", dataset.id);
                              } else if (dataset.kind === "benchmark") {
                                setEvaluationForm((current) => ({ ...current, benchmarkMode: "existing", benchmarkDatasetId: dataset.id }));
                                setActiveScreen("evaluation");
                                return;
                              } else if (dataset.kind === "corpus") {
                                setActiveScreen("evaluation");
                                return;
                              }
                              setActiveScreen("training");
                            }}
                          >
                            {dataset.kind === "corpus" ? "Use for RAG" : dataset.kind === "benchmark" ? "Use in eval" : "Use in run"}
                          </button>
                          <button
                            type="button"
                            className="btn btn-outline"
                            onClick={() => handleDownloadDataset(dataset)}
                            disabled={downloadingDatasetId === dataset.id}
                          >
                            {downloadingDatasetId === dataset.id ? "Downloading..." : "Download"}
                          </button>
                          {dataset.kind === "corpus" && !dataset.vectorStore?.collectionName ? (
                            <button
                              type="button"
                              className="btn btn-outline"
                              onClick={() => handleIngestCorpusDataset(dataset)}
                              disabled={ingestingDatasetId === dataset.id}
                            >
                              {ingestingDatasetId === dataset.id ? "Ingesting..." : "Ingest to pgvector"}
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="btn btn-danger"
                            onClick={() => handleDeleteDataset(dataset)}
                            disabled={deletingDatasetId === dataset.id}
                          >
                            {deletingDatasetId === dataset.id ? "Removing..." : "Remove"}
                          </button>
                        </div>
                      </div>
                    ))
                  ) : (
                    <EmptyState text="No datasets have been uploaded yet. Add QA and instruction datasets here before preparing a run." />
                  )}
                </div>
              </Card>
            </div>
          </section>

          {isBootLoading ? <div className="loading-strip">Connecting to DALTP API...</div> : null}
        </main>
      </div>
      <NotificationCenter toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}

function AuthScreen({ authMode, authForm, authError, authWorking, isBootLoading, onModeChange, onFieldChange, onSubmit }) {
  return (
    <div className="auth-shell">
      <div className="auth-panel">
        <div className="auth-brand">DALTP</div>
        <div className="auth-title">
          Domain-adaptive workflows, <em>with scoped access</em>
        </div>
        <div className="auth-subtitle">
          Sign in to access your own datasets, prepared bundles, and DALTP workspace history.
        </div>

        <div className="auth-toggle">
          <button
            type="button"
            className={`auth-toggle-btn ${authMode === "login" ? "active" : ""}`}
            onClick={() => onModeChange("login")}
          >
            Sign in
          </button>
          <button
            type="button"
            className={`auth-toggle-btn ${authMode === "register" ? "active" : ""}`}
            onClick={() => onModeChange("register")}
          >
            Create account
          </button>
        </div>

        <form className="auth-form" onSubmit={onSubmit}>
          {authMode === "register" ? (
            <FormField label="Full name">
              <input
                className="text-input"
                value={authForm.name}
                onChange={(event) => onFieldChange("name", event.target.value)}
                placeholder="Aditi Sharma"
              />
            </FormField>
          ) : null}

          <FormField label="Email">
            <input
              className="text-input"
              type="email"
              value={authForm.email}
              onChange={(event) => onFieldChange("email", event.target.value)}
              placeholder="you@example.com"
            />
          </FormField>

          <FormField label="Password">
            <input
              className="text-input"
              type="password"
              value={authForm.password}
              onChange={(event) => onFieldChange("password", event.target.value)}
              placeholder="Minimum 8 characters"
            />
          </FormField>

          {authError ? <div className="form-message error">{authError}</div> : null}

          <div className="form-actions">
            <button type="submit" className="btn btn-primary" disabled={authWorking || isBootLoading}>
              {authWorking || isBootLoading
                ? authMode === "register"
                  ? "Creating account..."
                  : "Signing in..."
                : authMode === "register"
                  ? "Create account"
                  : "Sign in"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function Topbar({ activeScreen, onNavigate, selectedRun, apiStatus, user, onLogout }) {
  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="brand-icon">D</div>
        <div className="brand-name">
          DALTP <span>Editorial</span>
        </div>
      </div>

      <nav className="topbar-nav">
        {topNavItems.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-link ${activeScreen === item.id ? "active" : ""}`}
            onClick={() => onNavigate(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      <div className="topbar-right">
        <div className="top-run-info">
          <span className={`pulse ${apiStatus === "offline" ? "pulse-offline" : ""}`} />
          <span className="pulse-label">
            {apiStatus === "connected" ? "API connected" : apiStatus === "offline" ? "API offline" : "Connecting"}
          </span>
        </div>

        {selectedRun ? (
          <div className="top-run-caption">
            <strong>{selectedRun.name}</strong>
            <span>{selectedRun.metricLabel}</span>
          </div>
        ) : null}

        {user ? (
          <div className="user-chip">
            <div className="user-chip-copy">
              <strong>{user.name}</strong>
            </div>
            <button type="button" className="btn btn-outline btn-tight" onClick={onLogout}>
              Logout
            </button>
          </div>
        ) : null}
      </div>
    </header>
  );
}

function Sidebar({ activeScreen, onNavigate, runningJobs, onOpenJobs }) {
  return (
    <aside className="sidebar">
      {sidebarSections.map((section) => (
        <div key={section.title}>
          <div className="sidebar-section">{section.title}</div>
          {section.items.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`sidebar-item ${activeScreen === item.id ? "active" : ""}`}
              onClick={() => onNavigate(item.id)}
            >
              <span className={`si-dot ${item.dot}`} />
              <span className="si-label">{item.label}</span>
            </button>
          ))}
        </div>
      ))}

      <div className="sidebar-activity">
        <div className="sidebar-section">Activity</div>
        {runningJobs.length ? (
          <button type="button" className="sidebar-activity-card" onClick={onOpenJobs}>
            <strong>{runningJobs.length} running job{runningJobs.length > 1 ? "s" : ""}</strong>
            <span>{runningJobs[0].title}</span>
          </button>
        ) : (
          <div className="sidebar-activity-card quiet">
            <strong>No active jobs</strong>
            <span>New generation, training, and evaluation work will appear here.</span>
          </div>
        )}
      </div>
    </aside>
  );
}

function PageHeader({ breadcrumb, title, emphasized, subtitle }) {
  return (
    <div className="page-header">
      <div className="breadcrumb">{breadcrumb}</div>
      <div className="page-title">
        {title} <em>{emphasized}</em>
      </div>
      <div className="page-sub">{subtitle}</div>
    </div>
  );
}

function Card({ title, subtitle, action, children }) {
  return (
    <div className="card">
      <div className="card-head">
        <div>
          <div className="card-title">{title}</div>
          {subtitle ? <div className="card-subtitle">{subtitle}</div> : null}
        </div>
        {action ? <div className="card-action">{action}</div> : null}
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function KpiCell({ stat }) {
  return (
    <div className="kpi-cell">
      <div className="kpi-eyebrow">{stat.label}</div>
      <div className={`kpi-number ${stat.tone === "green" ? "good" : stat.tone === "amber" ? "warn" : ""}`}>
        {stat.value}
      </div>
      <div className="kpi-delta">{stat.subtext}</div>
    </div>
  );
}

function EvalBarGroup({ label, metrics }) {
  return (
    <div className="eval-group">
      <div className="eval-group-title">{label}</div>
      <ThresholdRow label="ROUGE-L" value={metrics?.["ROUGE-L"]} tone="blue" />
      <ThresholdRow label="BERTScore F1" value={metrics?.["BERTScore F1"]} tone="green" />
      <ThresholdRow label="Fact Coverage" value={metrics?.["Fact Coverage"]} tone="amber" />
    </div>
  );
}

function ThresholdRow({ label, value, tone }) {
  const width = `${Math.max(4, Math.min(100, Number(value ?? 0) * 100))}%`;
  return (
    <div className="eval-row compact">
      <div className="eval-dim">{label}</div>
      <div className="eval-track">
        <div className={`eval-fill fill-${tone}`} style={{ width }} />
      </div>
      <div className="eval-num">{formatMetric(value)}</div>
    </div>
  );
}

function StatusPill({ status }) {
  const normalized = String(status).toLowerCase();
  return <span className={`status-pill ${normalized}`}>{status}</span>;
}

function ActivityBanner({ jobs, onOpenDatasets, onOpenEvaluation }) {
  const headline = jobs.length === 1 ? jobs[0].title : `${jobs.length} DALTP jobs are currently running`;
  return (
    <div className="activity-banner">
      <div className="activity-banner-copy">
        <strong>{headline}</strong>
        <span>{jobs.map((job) => prettifyLabel(job.type)).join(" · ")}</span>
      </div>
      <div className="activity-banner-actions">
        <button type="button" className="btn btn-outline btn-tight" onClick={onOpenDatasets}>
          Open datasets
        </button>
        <button type="button" className="btn btn-primary btn-tight" onClick={onOpenEvaluation}>
          Open evaluation
        </button>
      </div>
    </div>
  );
}

function JobPanel({ jobs, selectedJob, onSelectJob, emptyText }) {
  if (!jobs.length) {
    return <EmptyState text={emptyText || "No background jobs yet. Dataset generation, local training, and evaluation runs will appear here."} />;
  }

  return (
    <div className="job-panel">
      <div className="job-list">
        {jobs.map((job) => (
          <button
            key={job.id}
            type="button"
            className={`bundle-item ${selectedJob?.id === job.id ? "selected" : ""}`}
            onClick={() => onSelectJob(job.id)}
          >
            <div className="bundle-item-main">
              <strong>{job.title}</strong>
              <span>{prettifyLabel(job.type)}</span>
            </div>
            <StatusPill status={job.status} />
          </button>
        ))}
      </div>

      {selectedJob ? (
        <div className="job-detail">
          <dl className="detail-list">
            <div>
              <dt>Job type</dt>
              <dd>{prettifyLabel(selectedJob.type)}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd><StatusPill status={selectedJob.status} /></dd>
            </div>
          </dl>
          {selectedJob.error ? <div className="form-message error">{selectedJob.error}</div> : null}
          {selectedJob.result?.runId ? (
            <div className="field-empty-note">Evaluation run `{selectedJob.result.runId}` is now available in the reports view.</div>
          ) : null}
          <pre className="command-box">{selectedJob.logs || "No logs yet."}</pre>
        </div>
      ) : null}
    </div>
  );
}

function BundleSummary({ bundle, onDownload, downloading }) {
  return (
    <div className="bundle-summary">
      <div className="question-card">
        <div className="config-label">Bundle</div>
        <p>{bundle.runName}</p>
      </div>
      <div className="question-card">
        <div className="config-label">Execution mode</div>
        <p>{prettifyLabel(bundle.executionMode)}</p>
      </div>
      <div className="question-card">
        <div className="config-label">Download</div>
        <p>
          <button type="button" className="text-link-button" onClick={() => onDownload(bundle)} disabled={downloading}>
            {downloading ? "Downloading..." : "Fetch the zip bundle"}
          </button>{" "}
          and move it into the user's local DALTP repo or a Colab session.
        </p>
      </div>
    </div>
  );
}

function CommandPanel({ title, commands }) {
  return (
    <div className="command-panel">
      <div className="config-label">{title}</div>
      <pre className="command-box">{commands.length ? commands.join("\n") : "No commands available yet."}</pre>
    </div>
  );
}

function FormField({ label, children }) {
  return (
    <div className="form-field">
      <div className="config-label">{label}</div>
      {children}
    </div>
  );
}

function EmptyState({ text }) {
  return <div className="empty-state">{text}</div>;
}

function NotificationCenter({ toasts, onDismiss }) {
  if (!toasts.length) {
    return null;
  }

  const activeToast = toasts[0];

  return (
    <div className="notice-overlay">
      <div className={`notice-card ${activeToast.tone}`}>
        <div className="notice-kicker">{activeToast.tone === "error" ? "Action failed" : "Action completed"}</div>
        <div className="notice-text">{activeToast.text}</div>
        <div className="notice-actions">
          <button type="button" className="btn btn-primary btn-tight" onClick={() => onDismiss(activeToast.id)}>
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

function CustomSelect({ value, options, onChange, placeholder = "Select an option" }) {
  const [isOpen, setIsOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    function handlePointerDown(event) {
      if (rootRef.current && !rootRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  const selectedOption = options.find((option) => option.value === value);

  return (
    <div className={`custom-select ${isOpen ? "open" : ""}`} ref={rootRef}>
      <button
        type="button"
        className="custom-select-trigger"
        onClick={() => setIsOpen((current) => !current)}
        aria-expanded={isOpen}
      >
        <span>{selectedOption?.label ?? placeholder}</span>
        <span className="custom-select-chevron">v</span>
      </button>

      {isOpen ? (
        <div className="custom-select-menu">
          {options.map((option) => (
            <button
              key={`${option.value}-${option.label}`}
              type="button"
              className={`custom-select-option ${option.value === value ? "selected" : ""}`}
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

async function safeErrorMessage(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload.detail) && payload.detail.length) {
      const first = payload.detail[0];
      if (first?.msg) {
        const field = Array.isArray(first.loc) ? first.loc[first.loc.length - 1] : "field";
        return `${prettifyLabel(field)}: ${first.msg}`;
      }
    }
    return "Request failed.";
  } catch (error) {
    return "Request failed.";
  }
}

async function buildUploadFilePayload(entry) {
  return {
    name: entry.file.name,
    size: entry.file.size,
    mimeType: entry.file.type,
    relativePath: entry.relativePath,
    encoding: "base64",
    content: await fileToBase64(entry.file),
  };
}

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return btoa(binary);
}

function prettifyLabel(value) {
  return String(value ?? "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function slugifyValue(value) {
  return (
    String(value ?? "")
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "daltp-corpus"
  );
}

function inferDatasetDownloadName(dataset) {
  const safeName =
    String(dataset?.name ?? "dataset")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9._-]+/g, "-")
      .replace(/^-+|-+$/g, "") || "dataset";
  const extensionMap = {
    qa: ".jsonl",
    instruction: ".jsonl",
    benchmark: ".jsonl",
    corpus: ".jsonl",
  };
  return `${safeName}${extensionMap[dataset?.kind] ?? ".jsonl"}`;
}

function normalizedClientName(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter(Boolean)
    .join(" ");
}

function formatMetric(value) {
  if (typeof value !== "number") {
    return "--";
  }
  return value.toFixed(4);
}

function formatBytes(size) {
  if (!size) {
    return "0 B";
  }
  if (size < 1024) {
    return `${size} B`;
  }
  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${(size / (1024 * 1024)).toFixed(2)} MB`;
}

export default App;

