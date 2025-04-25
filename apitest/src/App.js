//App.js
import React, { useState, useEffect,useRef  } from "react";
import "./App.css";
import { Button } from "reactstrap";
import { Doughnut } from "react-chartjs-2";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import FileUploadSection from "./FileUploadSection";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
ChartJS.register(ArcElement, Tooltip, Legend);

const codeString = `import io.restassured.RestAssured;
import io.restassured.response.Response;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PetStoreApiTest {
    private static final String BASE_URL = "https://petstore.swagger.io/v2";

    @Test
    public void testGetPetById() {
        int petId = 1; // Replace with an existing pet ID
        Response response = RestAssured.given()
                .baseUri(BASE_URL)
                .when()
                .get("/pet/" + petId)
                .then()
                .extract().response();

        System.out.println(response.getBody().asString());
        Assert.assertEquals(response.getStatusCode(), 200, "Status code should be 200");
        Assert.assertTrue(response.getBody().asString().contains("\"id\":" + petId), "Response should contain pet ID");
    }
}`;

function App() {
  const [specUrl, setSpecUrl] = useState("");
  const [specFile, setSpecFile] = useState("");
  const [testCases, setTestCases] = useState({
    llm: [],
    ruleBased: [],
  });
  const [fileType, setFileType] = useState(null);
  const [testCaseSource, setTestCaseSource] = useState(null);
  const [selectedCases, setSelectedCases] = useState([]);
  const [file, setFile] = useState(null);
  const fileInputRef = useRef(null);
  const [cardData, setCardData] = useState({});
  const [submittedData, setSubmittedData] = useState([]);
  const [expandedCards, setExpandedCards] = useState({});
  const [selectedFile, setSelectedFile] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [chartData, setChartData] = useState({
    labels: ["Selected", "Remaining"],
    datasets: [
      {
        label: "Test Cases",
        data: [0, 100],
        backgroundColor: ["rgb(13, 184, 39)", "rgb(255, 99, 132)"],
        hoverOffset: 4,
      },
    ],
  });
  const data = {
    labels: [],
    datasets: [
      {
        label: "My First Dataset",
        data: [100, 10],
        backgroundColor: ["rgb(255, 99, 132)", "rgb(13, 184, 39)"],
        hoverOffset: 4,
      },
    ],
  };
  console.log(expandedCards);

  const config = {
    type: "doughnut",
    data: data,
  };
  const totalTestCases = testCases.llm.length + testCases.ruleBased.length;
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState(null);

  const allowedFileTypes = {
    "application/json": "JSON",
    "text/plain": "Text",
    "application/pdf": "PDF",
    "application/msword": "DOC",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
      "DOCX",
    "text/markdown": "MD",
  };

  useEffect(() => {
    const selectedPercentage = (selectedCases.length / totalTestCases) * 100;
    const remainingPercentage = 100 - selectedPercentage;
    setChartData({
      labels: ["Selected", "Remaining"],
      datasets: [
        {
          label: "Test Cases",
          data: [selectedPercentage, remainingPercentage],
          backgroundColor: ["rgb(13, 184, 39)", "rgb(255, 99, 132)"],
          hoverOffset: 4,
        },
      ],
    });
  }, [selectedCases, totalTestCases]);
  const handleExpand = (index, type) => {
    setExpandedCards((prev) => ({
      ...prev,
      [`${type}-${index}`]: !prev[`${type}-${index}`],
    }));
  };
  const handleCheckboxChange = (description, checked, index, type) => {
    setSelectedCases((prev) =>
      checked
        ? [...prev, description]
        : prev.filter((item) => item !== description)
    );
    handleExpand(index, type);
  };

  const handleFileUpload = (index, file) => {
    setCardData((prev) => ({
      ...prev,
      [index]: {
        ...prev[index],
        file: file,
      },
    }));
  };
console.log(selectedFile);

  const handleRunTest = (testDescription) => {
    console.log("Executing test:", testDescription);
    // Add actual test execution logic here
    alert(`Running test: ${testDescription}`);
  };
  const handleFileDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && allowedFileTypes[file.type]) {
      setSelectedFile(file);
      setSelectedFileName(file.name); // Add this line
      setSpecUrl("");
    } else {
      setSelectedFile(null);
      setSelectedFileName(null); // Add this line
      alert("Invalid file type...");
    }
    setIsDragging(false);
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && allowedFileTypes[file.type]) {
      setSelectedFile(file);
      setSelectedFileName(file.name);
      setSpecUrl("");
      console.log("Selected");
    } else {
      setSelectedFile(null);
      setSelectedFileName(null);
      alert("Please select...");
    }
  };

  const handleCancelFile = (e) => {
    e.stopPropagation();
    setSelectedFile(null);
    setSelectedFileName(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = ""; // Clear file input manually
    }
  };

  const handleDragDropClick = () => {
    document.getElementById("file-input").click();
  };

  const CardContent = ({
    inputCount,
    fileUploadCount,
    description,
    onSubmit,
  }) => {
    const [inputValues, setInputValues] = useState(Array(inputCount).fill(""));

    const handleInputChange = (index, value) => {
      setCardData((prev) => ({
        ...prev,
        [index]: {
          ...prev[index],
          params: value,
        },
      }));
    };

    const handleSubmit = () => {
      onSubmit(description, inputValues);
    };

    return (
      <div className="card-content">
        {Array.from({ length: inputCount }, (_, index) => (
          <input
            key={`input-${index}`}
            type="text"
            id={`spec-urls-${index}`}
            placeholder="Enter"
            value={inputValues[index]}
            onChange={(e) => handleInputChange(index, e.target.value)}
          />
        ))}
        {Array.from({ length: fileUploadCount }, (_, index) => (
          <FileUploadSection key={`file-upload-${index}`} />
        ))}
        <div className="buttons1">
          <button className="view1" onClick={handleSubmit}>
            Submit
          </button>
          <button className="view2" onClick={handleSubmit}>
            Send to TestRail➡️
          </button>
        </div>
      </div>
    );
  };

  const handleInputChange = (cardId, inputIndex, value) => {
    console.log(`Card ${cardId}, Input ${inputIndex}:`, value);
    setCardData((prev) => ({
      ...prev,
      [cardId]: {
        ...prev[cardId],
        inputs: {
          ...(prev[cardId]?.inputs || {}),
          [inputIndex]: value,
        },
      },
    }));
  };

  const handleFileUploadChange = (cardId, fileIndex, file) => {
    setCardData((prev) => ({
      ...prev,
      [cardId]: {
        ...prev[cardId],
        files: {
          ...(prev[cardId]?.files || {}),
          [fileIndex]: file,
        },
      },
    }));
  };

  const handleSubmit = (description, inputValues) => {
    setSubmittedData((prev) => [...prev, { description, inputValues }]);
    console.log("Submitted Data:", { description, inputValues });
  };

  // Update the card rendering to show additional test case details
  // Updated renderTestCases function in App.js
  // Updated renderTestCases function with validation
  function renderTestCases(testCases, type) {
    return testCases.map((fullTest, index) => {
      // Ensure fullTest is a string before processing
      const testString =
        typeof fullTest === "string" ? fullTest : JSON.stringify(fullTest);

      // Clean up test case formatting
      const description = testString
        .replace(/^(Verify|Test|Validate)\s+/i, "")
        .replace(/[^a-zA-Z0-9 ]/g, "")
        .trim();

      // Enhanced category detection logic
      const isPhysics =
        testString.toLowerCase().includes("physics") ||
        testString.toLowerCase().includes("gravity");
      const isGameplay =
        testString.toLowerCase().includes("mechanic") ||
        testString.toLowerCase().includes("control");
      const isAI =
        testString.toLowerCase().includes("ai") ||
        testString.toLowerCase().includes("pathfinding") ||
        testString.toLowerCase().includes("npc");
      const isUIUX =
        testString.toLowerCase().includes("menu") ||
        testString.toLowerCase().includes("hud") ||
        testString.toLowerCase().includes("navigation");
      const isMultiplayer =
        testString.toLowerCase().includes("network") ||
        testString.toLowerCase().includes("latency");
      const isValidation = testString.toLowerCase().includes("validate");

      const cardStyle = {
        borderLeft: isPhysics
          ? "4px solid #2196F3"
          : isGameplay
          ? "4px solid #4CAF50"
          : isAI
          ? "4px solid #9C27B0"
          : isUIUX
          ? "4px solid #FF9800"
          : isMultiplayer
          ? "4px solid #FF5722"
          : isValidation
          ? "4px solid #795548"
          : "4px solid #9e9e9e",
      };

      return (
        <div key={`${type}-${index}`} className="card" style={cardStyle}>
          <div className="card-header">
            <input
              type="checkbox"
              checked={selectedCases.includes(testString)}
              onChange={(e) =>
                handleCheckboxChange(testString, e.target.checked, index, type)
              }
            />
            <div className="description-container">
              <span className="description">{description}</span>
            </div>
            <div className="buttons">
              {testCaseSource === "url" ? (
                <>
                  <button className="test">Test</button>
                  <button className="playwright">Playwright</button>
                  <button className="selenium">Selenium</button>
                </>
              ) : (
                ""
              )}
            </div>
          </div>
          {expandedCards[`${type}-${index}`] && (
            <div className="card-content">
              {testCaseSource === "url" && (
                <>
                  <input
                    type="text"
                    placeholder="Enter test parameters"
                    className="test-input"
                    value={cardData[index]?.params || ""}
                    onChange={(e) => handleInputChange(index, e.target.value)}
                  />
                  <FileUploadSection
                    onFileUpload={(file) => handleFileUpload(index, file)}
                  />
                </>
              )}
              <button
                className="testrail"
                onClick={() => handleRunTest(testString)}
              >
                {testCaseSource === "url"
                  ? "Execute Test"
                  : "Send to TestRail➡️"}
              </button>
            </div>
          )}
        </div>
      );
    });
  }

  console.log(submittedData);

  const generateTestCases = async () => {
    setExpandedCards({});
    setSelectedFile(null);
    try {
      setIsGenerating(true); // Start loading
      let response;

      if (specUrl) {
        setTestCaseSource("url");
        response = await fetch("http://localhost:8081/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ spec_url: specUrl }),
        });
      } else if (selectedFile) {
        setTestCaseSource("document");
        const formData = new FormData();
        formData.append("file", selectedFile);
        response = await fetch("http://localhost:8081/generate", {
          method: "POST",
          body: formData,
        });
      } else {
        alert("Please provide either a URL or upload a file");
        return;
      }

      if (!response.ok)
        throw new Error(`HTTP error! Status: ${response.status}`);

      const data = await response.json();
      console.log(data);
      
      setTestCases({
        llm: data.llm_descriptions || [],
        ruleBased: data.rule_based_descriptions || [],
      });
    } catch (error) {
      console.error("API Error:", error);
      setTestCases({
        llm: [`Error: ${error.message}`],
        ruleBased: [],
      });
    } finally {
      setIsGenerating(false); // End loading regardless of success/error
    }
  };
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          backgroundColor: "#1e293b",
          alignItems: "center",
        }}
      >
        <div className="title">
          <h1>AIM 1.0</h1>
          {/* <h4>Testcase Generator</h4> */}
        </div>
      </div>
      <div style={{ display: "flex", justifyContent: "center", backgroundColor:"#0f172a"}}>
        <div className="input-container">
          <div className="url-input">
            <input
              type="text"
              id="spec-url"
              placeholder="Enter OpenAPI spec URL (e.g., https://petstore.swagger.io/v2/swagger.json)"
              value={specUrl}
              onChange={(e) => setSpecUrl(e.target.value)}
            />
          </div>
          <div className="drag-drop-container">
            <div
              className={`drag-drop-box ${isDragging ? "dragging" : ""} ${
                selectedFileName ? "has-file" : ""
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleFileDrop}
              onClick={handleDragDropClick}
            >
              {selectedFileName ? (
                <div className="selected-file">
                  <span className="file-name">{selectedFileName}</span>
                  <button
                    className="cancel-file-btn"
                    onClick={handleCancelFile}
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <p>Drag & Drop OpenAPI spec or document file here</p>
              )}
              <input
                type="file"
                id="file-input"
                accept=".json,.txt,.doc,.docx,.pdf,.md"
                onChange={handleFileSelect}
                ref={fileInputRef}
                style={{ display: "none" }}
              />
            </div>
          </div>
          <button className="generate-btn" onClick={generateTestCases}>
            Generate
          </button>
        </div>
      </div>
      <div style={{ display: "flex", width: "100%", justifyContent:"center", backgroundColor:"#0f172a"}}>
        <div className="container" style={{width:"100%"}}>
          <div className="item-list">
            {isGenerating ? (
              <div className="loading-container">
                <div className="loading-text">Generating Test Cases...</div>
              </div>
            ) : (
              <div id="llm-test-cases">
                {renderTestCases(testCases.llm, "llm")}
                {renderTestCases(testCases.ruleBased, "rule")}
              </div>
            )}
          </div>
        </div>
        {testCaseSource === "url" ? (
          <div className="piechart">
            <div className="sidecontainer">
              <p style={{ margin: "0 0 0 50%" }}>SCRIPT</p>
              <div className="pieitem1">
                <div className="code-scroll">
                  <SyntaxHighlighter language="java" style={oneDark}>
                    {codeString}
                  </SyntaxHighlighter>
                </div>
              </div>
            </div>
            <div className="sidecontainer1">
              <div className="pieitem">
                Open
                <Doughnut
                  data={chartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      tooltip: {
                        callbacks: {
                          label: (context) => {
                            if (context.label === "Selected")
                              return `Selected Test Cases: ${selectedCases.length}`;
                            if (context.label === "Remaining")
                              return `Remaining Test Cases: ${
                                totalTestCases - selectedCases.length
                              }`;
                            return "";
                          },
                        },
                      },
                    },
                  }}
                  className="doughnut"
                />
              </div>
              <div className="pieitem">
                Pass
                <Doughnut data={data} className="doughnut" />
              </div>
              <div className="pieitem">
                Fail
                <Doughnut data={data} className="doughnut" />
              </div>
            </div>
          </div>
        ) : (
          ""
        )}
      </div>
    </div>
  );
}

export default App;
