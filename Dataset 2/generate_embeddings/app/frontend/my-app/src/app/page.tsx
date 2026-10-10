"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const router = useRouter();

  const toggleModel = (model: string) => {
    setSelectedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    );
  };

  const isSelected = (model: string) => selectedModels.includes(model);

  const handleStart = () => {
    const params = new URLSearchParams();
    selectedModels.forEach((model) => params.append("models", model));
    router.push(`/recommend?${params.toString()}`);
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-white">
      <div className="bg-gray-100 bg-opacity-70 border border-black rounded-2xl p-10 w-full max-w-2xl flex flex-col items-center">
        <h1 className="text-2xl text-center mb-6">
          Welcome to my Third Year Project demo. To start, choose the models you want to see recommendations for and then click the start button.
        </h1>

        <h2 className="text-3xl font-bold mb-6">Model Choice</h2>

        <div className="flex flex-col space-y-4 w-64">
          <button
            onClick={() => toggleModel("dinov2")}
            className={`w-full px-6 py-3 rounded-2xl border text-black ${
              isSelected("dinov2")
                ? "bg-[#fff4c2] border-[#c1a200]"
                : "bg-transparent border-black"
            }`}
          >
            DINOv2
          </button>
          <button
            onClick={() => toggleModel("clip-image")}
            className={`w-full px-6 py-3 rounded-2xl border text-black ${
              isSelected("clip-image")
                ? "bg-[#ffcccc] border-[#990000]"
                : "bg-transparent border-black"
            }`}
          >
            CLIP (Image)
          </button>
          <button
            onClick={() => toggleModel("qwen")}
            className={`w-full px-6 py-3 rounded-2xl border text-black ${
              isSelected("qwen")
                ? "bg-[#d9ead3] border-[#38761d]"
                : "bg-transparent border-black"
            }`}
          >
            Qwen
          </button>
          <button
            onClick={() => toggleModel("clip-text")}
            className={`w-full px-6 py-3 rounded-2xl border text-black ${
              isSelected("clip-text")
                ? "bg-[#fce5cd] border-[#b45f06]"
                : "bg-transparent border-black"
            }`}
          >
            CLIP (Text)
          </button>
        </div>

        <button
          onClick={handleStart}
          disabled={selectedModels.length === 0}
          className={`mt-10 w-60 px-6 py-4 rounded-xl border text-black bg-gradient-to-r from-[#cfe2f3] to-[#a4c2f4] border-[#3c78d8] text-lg font-bold shadow-md ${
            selectedModels.length === 0 ? "opacity-40 cursor-not-allowed" : "hover:brightness-105"
          }`}
        >
          Start
        </button>
      </div>
    </main>
  );
}
