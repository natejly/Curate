import useSWR from "swr";
import { useState } from "react";
import { useRouter } from "next/navigation";

export function AvailableDatasetsSection() {
  const fetcher = (url: string) => fetch(url).then(res => res.json());
  const { data, error, isLoading } = useSWR("http://localhost:8000/available-datasets", fetcher);
  const [selected, setSelected] = useState<string | null>(null);
  const [trainStatus, setTrainStatus] = useState<string>("");
  const router = useRouter();

  async function handleTrain() {
    if (!selected) return;
    setTrainStatus("Training started...");
    const sessionId = crypto.randomUUID();
    setSelected(null); // Deselect after training
    router.push(`/training-console?sessionId=${encodeURIComponent(sessionId)}`);
    // Fire and forget backend request
    fetch(`http://localhost:8000/train-s3/${encodeURIComponent(selected)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
  }

  return (
    <div className="w-full max-w-xl mx-auto mt-10 p-6 rounded-2xl bg-neutral-900 border border-neutral-400 text-neutral-100">
      <div className="text-lg font-bold mb-4">Available Datasets (Ready to Train)</div>
      {isLoading && <div>Loading datasets...</div>}
      {error && <div className="text-red-400">Error loading datasets.</div>}
      {data && data.datasets && data.datasets.length > 0 ? (
        <ul className="divide-y divide-neutral-700">
          {data.datasets.map((name: string) => {
            const displayName = name.replace(/\.zip$/i, "");
            const isSelected = selected === name;
            return (
              <li
                key={name}
                className={`flex items-center justify-between py-3 px-2 cursor-pointer transition-all duration-150 rounded-lg ${isSelected ? 'bg-neutral-700/40' : 'hover:bg-neutral-800/60'}`}
                onClick={() => setSelected(isSelected ? null : name)}
              >
                <div className="flex items-center gap-3 flex-1">
                  <span className="text-base font-semibold text-neutral-100">{displayName}</span>
                </div>
                <span className="bg-neutral-600 text-neutral-200 text-xs px-2 py-1 rounded-full ml-2">Image Classification</span>
              </li>
            );
          })}
        </ul>
      ) : (
        <div className="text-neutral-400">No datasets found in S3.</div>
      )}
      {selected && (
        <div className="flex justify-center mt-6">
          <button
            className="bg-neutral-900 hover:bg-neutral-700 text-neutral-100 px-6 py-2 rounded-lg shadow text-lg font-semibold transition-all duration-150 border border-neutral-400"
            onClick={handleTrain}
            disabled={trainStatus === "Training started..."}
          >
            {trainStatus === "Training started..." ? "Training..." : "Train Selected Dataset"}
          </button>
        </div>
      )}
    </div>
  );
}
