import React, { useEffect, useState } from "react";
import "../styles/MVP.css";

const SavedResults = () => {
    const [saved, setSaved] = useState([]);

    useEffect(() => {
        const results = JSON.parse(localStorage.getItem("savedResults") || "[]");
        setSaved(results);
    }, []);

    const clearSaved = () => {
        localStorage.removeItem("savedResults");
        setSaved([]);
    };

    return (
        <div className="mvp-container">
            <div className="mvp-card" style={{ width: "100%" }}>
                <h2>Saved Results</h2>
                {saved.length === 0 ? (
                    <p>No saved results yet.</p>
                ) : (
                    <div className="saved-list">
                        {saved.map((r, i) => (
                            <div key={i} className="saved-item">
                                <img src={r.image} alt="MRI" className="uploaded-image" />
                                <div>
                                    <p>
                                        <strong>Prediction:</strong> {r.prediction}
                                    </p>
                                    <p>
                                        <strong>Explanation:</strong> {r.explanation}
                                    </p>
                                    <p>
                                        <em>{r.date}</em>
                                    </p>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
                {saved.length > 0 && (
                    <button className="diagnosis-button" onClick={clearSaved}>
                        🗑 Clear All
                    </button>
                )}
            </div>
        </div>
    );
};

export default SavedResults;
