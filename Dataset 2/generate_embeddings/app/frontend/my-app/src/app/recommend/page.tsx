"use client";
import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { FaChevronDown } from "react-icons/fa";
import Papa from "papaparse";

type LabelMap = {
  [key: string]: {
    title: string;
    desc: string;
    style: string;
    brand: string;
    condition: string;
    colour: string;
    gender: string;
    size: string;
  };
};


export default function RecommendPage() {
 
  const filterOptions = {
    size: ["XS", "S", "M", "L", "XL", "8 (UK)", "10 (UK)", "12 (UK)", "14 (UK)", "One Size"],
    brand: ["Marks & Spencer", "Primark", "Next", "Nike", "Other"],
    colour: ["Black", "Blue", "Grey", "White", "Green", "Multicoloured", "Beige", "Red", "Brown"],
    condition: ["Like New", "New", "Good", "Fair"],
    gender: ["Men", "Women"],
    category: [
      "Jackets",
      "Sweatshirt",
      "Trousers",
      "Straight Jeans",
      "Slim Jeans",
      "Skinny Jeans",
      "Printed T-Shirts",
      "Basic T-Shirts",
      "Polo Shirts",
      "Formal Shirts",
      "Blouses",
      "Coats",
      "Other"
    ]
  };
  
  const [proceedClicked, setProceedClicked] = useState(false);


  const searchParams = useSearchParams();
  const selectedModels = searchParams.getAll("models");

  const [openDropdown, setOpenDropdown] = useState<string | null>(null);
  const [selectedFilters, setSelectedFilters] = useState<Record<string, string[]>>({});
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);


  const [recommendedImagesDino, setDino] = useState<string[]>([]);
  const [recommendedImagesClip, setClip] = useState<string[]>([]);
  const [recommendedClipTitle, setClipTitle] = useState<string[]>([]);
  const [recommendedClipDesc, setClipDesc] = useState<string[]>([]);
  const [recommendedQwenTitle, setQwenTitle] = useState<string[]>([]);
  const [recommendedQwenDesc, setQwenDesc] = useState<string[]>([]);

  const [labels, setLabels] = useState<LabelMap>({});
  const [uploadedKey, setUploadedKey] = useState<string | null>(null);


  useEffect(() => {
    fetch("/labels.csv")
      .then((res) => res.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          header: true,
          skipEmptyLines: true,
          complete: (results: Papa.ParseResult<any>) => {
            const parsed: LabelMap = {};
            (results.data as any[]).forEach((row, index) => {
              const key = String(index + 1);
              parsed[key] = {
                title: row["Title"] || "",
                desc: row["Description"] || "",
                style: row["Style"] || "",
                brand: row["Brand"] || "",
                condition: row["Condition"] || "",
                colour: row["Colour"] || "",
                gender: row["Gender"] || "",
                size: row["Size"] || "",
              };
            });
            setLabels(parsed);
          },
        });
      })
      .catch((err) => console.error("Failed to parse labels.csv", err));
  }, []);

  useEffect(() => {
    if (!proceedClicked || !uploadedFile) return;
  
    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("filters", JSON.stringify(selectedFilters));
  
    const fetchAndSet = async (endpoint: string, setter: (data: string[]) => void) => {
      const res = await fetch(`http://localhost:8000/upload/${endpoint}`, {
        method: "POST",
        body: formData,
      });
      const json = await res.json();
      if (json.results) setter(json.results.map((n: string) => `/images/${n}`));
    };
  
    (async () => {
      if (selectedModels.includes("dinov2")) await fetchAndSet("dinov2", setDino);
      if (selectedModels.includes("clip-image")) await fetchAndSet("clip", setClip);
      if (selectedModels.includes("clip-text")) {
        await fetchAndSet("clip-title", setClipTitle);
        await fetchAndSet("clip-desc", setClipDesc);
      }
      if (selectedModels.includes("qwen")) {
        await fetchAndSet("qwen-title", setQwenTitle);
        await fetchAndSet("qwen-desc", setQwenDesc);
      }
    })();
  
  }, [proceedClicked, uploadedFile]);
  

  const toggleDropdown = (filter: string) => {
    setOpenDropdown(prev => (prev === filter ? null : filter));
  };

  const toggleOption = (filter: string, option: string) => {
    setSelectedFilters(prev => {
      const current = prev[filter] || [];
      const updated = current.includes(option)
        ? current.filter(o => o !== option)
        : [...current, option];
      return { ...prev, [filter]: updated };
    });
  };

  const isSelected = (filter: string) => (selectedFilters[filter]?.length ?? 0) > 0;

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
  
    setUploadedImage(URL.createObjectURL(file));
    setUploadedFile(file); // ← store file for future use
    const baseName = file.name.split(".")[0];
    setUploadedKey(baseName);
  };
  

  const getImageKey = (img: string): string => {
    return img.split("/").pop()?.split(".")[0] || "";
  };
  

 
  const uploadedLabel = uploadedKey ? labels[String(Number(uploadedKey) + 1)] : undefined;


  const [activePopupImage, setActivePopupImage] = useState<string | null>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!(e.target as HTMLElement).closest(".ppopup-content")) {
        setActivePopupImage(null);
      }
    };
    window.addEventListener("mousedown", handleClickOutside);
    return () => window.removeEventListener("mousedown", handleClickOutside);
  }, []);


  return (
    <main className="min-h-screen flex items-center justify-center bg-white px-6 py-8">
      <div className="bg-gray-100 bg-opacity-70 border border-black rounded-2xl p-10 w-full max-w-10xl flex flex-col items-center space-y-0 pb-0">
        <div className="space-y-6 flex flex-col items-center w-full">
          {/* Upload Section */}
          <div className="space-y-8 flex flex-col items-center w-full">
            <div className="flex flex-col items-center space-y-4 w-full">
              <h2 className="text-2xl font-bold text-center">Upload an image</h2>
              <label htmlFor="fileInput" className="w-60 px-6 py-4 text-center rounded-xl border text-black text-lg font-bold shadow-md cursor-pointer bg-gradient-to-r from-[#cfe2f3] to-[#a4c2f4] border-[#3c78d8] hover:brightness-105">
                Choose File
                <input id="fileInput" type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
              </label>
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-col items-center space-y-4 pt-2 w-full pb-10">
            <h2 className="text-2xl font-bold text-center">Filters</h2>
            <div className="flex justify-center gap-4 w-full flex-wrap">
              {Object.entries(filterOptions).map(([filter, options]) => (
                <div key={filter} className="relative">
                  <button
                    onClick={() => toggleDropdown(filter)}
                    className={`w-36 px-4 py-3 rounded-2xl border text-black flex items-center justify-between ${
                      isSelected(filter) ? "bg-[#e0f7fa] border-[#007c91]" : "bg-transparent border-black"
                    }`}
                  >
                    <span>{filter.charAt(0).toUpperCase() + filter.slice(1)}</span>
                    <FaChevronDown className="ml-2 text-sm" />
                  </button>
                  {openDropdown === filter && (
                    <div className="absolute z-10 mt-2 w-48 bg-white rounded-xl shadow-lg max-h-56 overflow-y-auto border border-gray-300">
                      <div className="p-3 space-y-2">
                        {options.map(option => (
                          <div key={option} className="flex justify-between items-center py-2 border-b">
                            <span className="font-medium">{option}</span>
                            <input
                              type="checkbox"
                              checked={selectedFilters[filter]?.includes(option) || false}
                              onChange={() => toggleOption(filter, option)}
                              className="w-5 h-5 rounded border-gray-400"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>


        {/* Proceed Button */}
        <div className="flex justify-center w-full mt-6 mb-6">
          <button
            onClick={() => setProceedClicked(true)}
            disabled={!uploadedImage}
            className={`w-72 px-8 py-3 text-center rounded-xl border text-black text-lg font-bold shadow-md transition mb-10
              ${
                uploadedImage
                  ? "bg-gradient-to-r from-[#b6e0b0] to-[#6fcf97] border-[#2e7d32] hover:brightness-105"
                  : "bg-gray-300 border-gray-400 cursor-not-allowed opacity-50"
              }`}
          >
            Proceed
          </button>
        </div>

        {/* Divider */}
        <div className="-mx-10 w-[calc(100%+5rem)]"><hr className="border-t border-black w-full" /></div>

        {/* Uploaded Image & Results */}
        <div className="-mx-10 w-[calc(100%+5rem)] space-y-0 flex flex-col gap-0 mb-0">
          {uploadedImage && (
            <div className="flex flex-col md:flex-row gap-6 items-center bg-[rgba(229,231,235,0.4)] p-4 w-full">
              <img src={uploadedImage} alt="Uploaded" className="w-48 h-64 rounded-xl object-cover" />
              <div className="text-left space-y-1">
                <p className="text-lg font-bold">{uploadedLabel?.title || "Title"}</p>
                <p className="text-sm italic text-gray-700">{uploadedLabel?.desc || "Description of the uploaded image will go here."}</p>
                {uploadedLabel && (
                  <>
                    <p className="text-sm"><strong>Style:</strong> {uploadedLabel.style}</p>
                    <p className="text-sm"><strong>Brand:</strong> {uploadedLabel.brand}</p>
                    <p className="text-sm"><strong>Condition:</strong> {uploadedLabel.condition}</p>
                    <p className="text-sm"><strong>Colour:</strong> {uploadedLabel.colour}</p>
                    <p className="text-sm"><strong>Gender:</strong> {uploadedLabel.gender}</p>
                    <p className="text-sm"><strong>Size:</strong> {uploadedLabel.size}</p>
                  </>
                )}
              </div>
            </div>
          )}

      

          {/* DINOv2 */}
          {selectedModels.includes("dinov2") && recommendedImagesDino.length > 0 && (
            <div className="bg-[rgba(254,243,199,0.4)] p-6 w-full mb-0">
              <div className="text-black text-xl font-semibold text-center mb-6">DINOv2 Recommendations</div>
              <div className="flex flex-nowrap overflow-x-auto gap-4">
                {recommendedImagesDino.map((img, idx) => (
                  <img
                    key={idx}
                    src={img}
                    className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                    alt={`DinoV2 ${idx}`}
                    onClick={() => setActivePopupImage(img)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* CLIP (Image) */}
          {selectedModels.includes("clip-image") && recommendedImagesClip.length > 0 && (
              <div className="bg-[rgba(255,204,204,0.4)] p-6 w-full mb-0">
                <div className="text-black text-xl font-semibold text-center mb-6">CLIP (Image) Recommendations</div>
                <div className="flex flex-nowrap overflow-x-auto gap-4">
                  {recommendedImagesClip.map((img, idx) => (
                    <img
                      key={idx}
                      src={img}
                      className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                      alt={`CLIP Image ${idx}`}
                      onClick={() => setActivePopupImage(img)}
                    />
                  ))}
                </div>
              </div>
            )}



          {/* CLIP (Text) */}
          {selectedModels.includes("clip-text") && (recommendedClipTitle.length > 0 || recommendedClipDesc.length > 0) && (
              <div className="bg-[rgba(255,200,150,0.4)] p-6 w-full mb-0">
                <div className="text-black text-xl font-semibold text-center mb-4">CLIP (Text) Recommendations</div>

                {recommendedClipTitle.length > 0 && (
                  <>
                    <p className="text-lg font-semibold mb-2">Title-based</p>
                    <div className="flex flex-nowrap overflow-x-auto gap-4 mb-4">
                      {recommendedClipTitle.map((img, idx) => (
                        <img
                          key={idx}
                          src={img}
                          className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                          alt={`CLIP Title ${idx}`}
                          onClick={() => setActivePopupImage(img)}
                        />
                      ))}
                    </div>
                  </>
                )}

                {recommendedClipDesc.length > 0 && (
                  <>
                    <p className="text-lg font-semibold mb-2">Description-based</p>
                    <div className="flex flex-nowrap overflow-x-auto gap-4">
                      {recommendedClipDesc.map((img, idx) => (
                        <img
                          key={idx}
                          src={img}
                          className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                          alt={`CLIP Desc ${idx}`}
                          onClick={() => setActivePopupImage(img)}
                        />
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}


              {/* Qwen */}
              {selectedModels.includes("qwen") && (recommendedQwenTitle.length > 0 || recommendedQwenDesc.length > 0) && (
                <div className="bg-[rgba(210,240,220,0.4)] p-6 w-full mb-0">
                  <div className="text-black text-xl font-semibold text-center mb-4">Qwen Recommendations</div>

                  {recommendedQwenTitle.length > 0 && (
                    <>
                      <p className="text-lg font-semibold mb-2">Title-based</p>
                      <div className="flex flex-nowrap overflow-x-auto gap-4 mb-4">
                        {recommendedQwenTitle.map((img, idx) => (
                          <img
                            key={idx}
                            src={img}
                            className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                            alt={`Qwen Title ${idx}`}
                            onClick={() => setActivePopupImage(img)}
                          />
                        ))}
                      </div>
                    </>
                  )}

                  {recommendedQwenDesc.length > 0 && (
                    <>
                      <p className="text-lg font-semibold mb-2">Description-based</p>
                      <div className="flex flex-nowrap overflow-x-auto gap-4">
                        {recommendedQwenDesc.map((img, idx) => (
                          <img
                            key={idx}
                            src={img}
                            className="w-40 h-60 object-cover rounded-xl cursor-pointer"
                            alt={`Qwen Desc ${idx}`}
                            onClick={() => setActivePopupImage(img)}
                          />
                        ))}
                      </div>
                    </>
                  )}
                </div>
              )}


          {activePopupImage && (
            <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-40 z-50">
              <div className="popup-content bg-white p-6 rounded-2xl shadow-xl w-[90%] max-w-md relative">
                <button
                  onClick={() => setActivePopupImage(null)}
                  className="absolute top-2 right-3 text-gray-500 hover:text-black text-lg"
                >
                  ×
                </button>
                {(() => {
                  const key = String(Number(getImageKey(activePopupImage)) + 1);

                  const label = labels[key];
                  return (
                    <>
                      <img src={activePopupImage} alt="Popup" className="w-full h-64 object-cover rounded-xl mb-4" />
                      <p className="text-lg font-bold">{label?.title || "No title"}</p>
                      <p className="text-sm italic text-gray-700 mb-2">{label?.desc || "No description available."}</p>
                      <div className="space-y-1 text-sm">
                        <p><strong>Style:</strong> {label?.style}</p>
                        <p><strong>Brand:</strong> {label?.brand}</p>
                        <p><strong>Condition:</strong> {label?.condition}</p>
                        <p><strong>Colour:</strong> {label?.colour}</p>
                        <p><strong>Gender:</strong> {label?.gender}</p>
                        <p><strong>Size:</strong> {label?.size}</p>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>
          )}

          {!recommendedImagesDino.length &&
           !recommendedImagesClip.length &&
           !recommendedClipTitle.length &&
           !recommendedClipDesc.length &&
           !recommendedQwenTitle.length &&
           !recommendedQwenDesc.length && (
            <div className="flex items-center justify-center p-6 w-full h-64 text-center rounded-xl">
              <p className="text-gray-600 text-xl italic">nothing to see here yet</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
