import React, { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { auth } from "../Firebase/config";
import "../styles/Navbar.css";

export default function Navbar() {
    const [menuOpen, setMenuOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);
    const [user, setUser] = useState(null);
    const location = useLocation();
    const navigate = useNavigate();

    // 🔥 Track Firebase auth state
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });

        const handleScroll = () => setScrolled(window.scrollY > 30);
        window.addEventListener("scroll", handleScroll);

        return () => {
            unsubscribe();
            window.removeEventListener("scroll", handleScroll);
        };
    }, []);

    // 🚪 Logout handler
    const handleLogout = async () => {
        await signOut(auth);
        alert("You have been logged out.");
        navigate("/");
    };

    return (
        <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
            <div className="nav-container">
                {/* 🔵 Logo Section */}
                <div className="nav-left">
                    <img
                        src="https://cdn-icons-png.flaticon.com/512/1065/1065694.png"
                        alt="logo"
                        className="nav-logo"
                    />
                    <h1 className="nav-title">Brain MRI Segmentation</h1>
                </div>

                {/* 🔗 Navigation Links */}
                <div className={`nav-links ${menuOpen ? "active" : ""}`}>
                    <Link
                        to="/"
                        onClick={() => setMenuOpen(false)}
                        className={location.pathname === "/" ? "active" : ""}
                    >
                        Home
                    </Link>

                    <Link
                        to="/mvp"
                        onClick={() => setMenuOpen(false)}
                        className={location.pathname === "/mvp" ? "active" : ""}
                    >
                        MVP
                    </Link>

                    <Link
                        to="/saved"
                        onClick={() => setMenuOpen(false)}
                        className={location.pathname === "/saved" ? "active" : ""}
                    >
                        Saved Results
                    </Link>

                    {/* 👤 Dynamic Login / Logout */}
                    {!user ? (
                        <Link
                            to="/login"
                            onClick={() => setMenuOpen(false)}
                            className={location.pathname === "/login" ? "active" : ""}
                        >
                            Login
                        </Link>
                    ) : (
                        <div className="user-info">
                            <span className="welcome-text">
                                Hi, {user.displayName?.split(" ")[0] || user.email.split("@")[0]}
                            </span>
                            <button className="logout-btn" onClick={handleLogout}>
                                Logout
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </nav>
    );
}
