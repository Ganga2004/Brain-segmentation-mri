import React from "react";
import "../styles/Home.css";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
 // rename uploaded image accordingly

const Home = () => {
    return (
        <div className="home-container">
            <motion.section
                className="hero"
                initial={{ opacity: 0, y: -40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 1 }}
            >
                <div className="hero-text">
                    <h1>AI-Powered Brain MRI Segmentation</h1>
                    <p>
                        Our platform leverages advanced AI techniques like Vision Transformers,
                        CNNs, and Cross-Scale Attention Fusion to accurately segment and label
                        brain MRI scans. Designed for researchers and clinicians, this system
                        enhances diagnostic accuracy, treatment planning, and research analysis.
                    </p>
                    <Link to="/mvp" className="hero-btn">
                        Try MVP 🚀
                    </Link>
                </div>

                <div className="hero-image">
                    <motion.img
                        src="src\assets\images\Brainimage.png"
                        alt="AI Brain MRI"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3, duration: 1 }}
                    />
                </div>
            </motion.section>

            <motion.section
                className="about"
                initial={{ opacity: 0, y: 40 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 1 }}
            >
                <h2>About the Project</h2>
                <p>
                    Brain MRI Segmentation plays a critical role in identifying abnormalities
                    such as tumors, lesions, and structural changes. Our hybrid model combines
                    local detail extraction of CNNs with the global attention of Transformers.
                    It aims to reduce manual workload, improve diagnostic precision, and make
                    segmentation accessible via an interactive web interface.
                </p>

                <div className="feature-cards">
                    <div className="feature-card">
                        <img src="https://cdn-icons-png.flaticon.com/512/2210/2210198.png" alt="Accuracy" />
                        <h3>High Accuracy</h3>
                        <p>Advanced ViT-based segmentation ensures pixel-perfect boundaries.</p>
                    </div>
                    <div className="feature-card">
                        <img src="https://cdn-icons-png.flaticon.com/512/2921/2921222.png" alt="Speed" />
                        <h3>Fast & Efficient</h3>
                        <p>Optimized model for real-time analysis and visualization.</p>
                    </div>
                    <div className="feature-card">
                        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="User" />
                        <h3>User-Friendly</h3>
                        <p>Simple upload interface for clinicians and researchers.</p>
                    </div>
                </div>
            </motion.section>
        </div>
    );
};

export default Home;
