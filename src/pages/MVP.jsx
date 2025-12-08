// src/pages/MVP.jsx
import React, { useState, useEffect } from "react";
import "../styles/MVP.css";
import { auth } from "../Firebase/config";
import { onAuthStateChanged } from "firebase/auth";
import { useNavigate } from "react-router-dom";

const MVP = () => {
    const [image, setImage] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const navigate = useNavigate();

    // 🔍 Check user login status
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            setIsLoggedIn(!!user);
        });
        return () => unsubscribe();
    }, []);

    // 🧠 Handle image upload
    const handleImageUpload = (event) => {
        if (!isLoggedIn) {
            alert("⚠️ Please log in before uploading images.");
            return;
        }

        const file = event.target.files[0];
        if (file) {
            setImage(URL.createObjectURL(file));
        }
    };

    // 🧠 Dummy chatbot (for now)
    const sendMessage = (text, sender = "user") => {
        setMessages((prev) => [...prev, { text, sender }]);
    };

    const handleDiagnosis = () => {
        if (!isLoggedIn) {
            alert("You must log in to analyze images.");
            return;
        }

        sendMessage("Analyzing MRI image... Please wait.", "bot");
        setTimeout(() => {
            sendMessage("✅ Diagnosis Complete: No abnormalities detected.", "bot");
        }, 1500);
    };

    // 💾 Save button only for logged-in users
    const handleSave = () => {
        if (!isLoggedIn) {
            alert("Please log in to save your results.");
            return;
        }
        alert("✅ Result saved successfully!");
    };

    return (
        <div className="mvp-container">
            {/* Image Upload Section */}
            <div className="mvp-card upload-section">
                <h2>🧠 Upload Brain MRI</h2>

                {!isLoggedIn ? (
                    <div className="login-prompt">
                        <p className="login-text">⚠️ Please log in to upload and analyze MRI scans.</p>
                        <button className="login-btn" onClick={() => navigate("/login")}>
                            Go to Login / Signup
                        </button>
                    </div>
                ) : (
                    <>
                        <div
                            className="upload-box"
                            onClick={() => document.getElementById("imageUpload").click()}
                        >
                            <input
                                type="file"
                                id="imageUpload"
                                accept="image/*"
                                hidden
                                onChange={handleImageUpload}
                            />
                            <p>📤 Click or drag & drop to upload MRI image</p>
                            {image && <img src={image} alt="Uploaded" className="uploaded-image" />}
                        </div>

                        <button className="diagnosis-button" onClick={handleDiagnosis}>
                            Analyze MRI 🧩
                        </button>

                        <button className="save-button" onClick={handleSave}>
                            Save Result 💾
                        </button>
                    </>
                )}
            </div>

            {/* Chatbot Section */}
            <div className="mvp-card chatbot-section">
                <h2>💬 AI Chat Assistant</h2>
                <div className="chatbot-messages">
                    {messages.map((msg, i) => (
                        <div key={i} className={`chatbot-message ${msg.sender}-message`}>
                            {msg.text}
                        </div>
                    ))}
                </div>

                {isLoggedIn ? (
                    <div className="chatbot-input">
                        <input
                            type="text"
                            placeholder="Ask something..."
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && e.target.value.trim()) {
                                    sendMessage(e.target.value);
                                    setTimeout(() => {
                                        sendMessage("🤖 AI is processing your question...", "bot");
                                    }, 1000);
                                    e.target.value = "";
                                }
                            }}
                        />
                        <button
                            onClick={() => {
                                const input = document.querySelector(".chatbot-input input");
                                if (input.value.trim()) {
                                    sendMessage(input.value);
                                    input.value = "";
                                }
                            }}
                        >
                            Send
                        </button>
                    </div>
                ) : (
                    <p className="login-text">Please log in to chat with AI.</p>
                )}
            </div>
        </div>
    );
};

export default MVP;
