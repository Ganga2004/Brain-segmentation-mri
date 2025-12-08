import React, { useState, useEffect } from "react";
import {
    createUserWithEmailAndPassword,
    signInWithEmailAndPassword,
    signInWithPopup,
    onAuthStateChanged,
    signOut,
} from "firebase/auth";
import { auth, googleProvider } from "../Firebase/config";
import "../styles/Login.css";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

const Login = () => {
    const [isSignup, setIsSignup] = useState(false);
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [user, setUser] = useState(null);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });
        return () => unsubscribe();
    }, []);

    // 🧠 Email/Password Auth
    const handleAuth = async () => {
        if (!email || !password) {
            toast.warning("Please enter both email and password!");
            return;
        }

        try {
            if (isSignup) {
                await createUserWithEmailAndPassword(auth, email, password);
                toast.success("🎉 Account created successfully!");
            } else {
                await signInWithEmailAndPassword(auth, email, password);
                toast.success("✅ Login successful!");
            }
        } catch (error) {
            console.error("Auth Error:", error.code);

            // 🧩 Smart error handling
            if (error.code === "auth/user-not-found") {
                toast.error("User does not exist. Please sign up first!");
            } else if (error.code === "auth/wrong-password") {
                toast.error("Incorrect password! Try again.");
            } else if (error.code === "auth/email-already-in-use") {
                toast.info("User already exists. Please log in instead.");
            } else if (error.code === "auth/invalid-email") {
                toast.error("Invalid email format!");
            } else {
                toast.error("Authentication failed. Please try again.");
            }
        }
    };

    // 🧠 Google Sign-In
    const handleGoogleLogin = async () => {
        try {
            await signInWithPopup(auth, googleProvider);
            toast.success("🌐 Logged in with Google!");
        } catch (error) {
            console.error("Google login error:", error.code);
            toast.error("Google Sign-In failed. Try again.");
        }
    };

    // 🧠 Logout
    const handleLogout = async () => {
        await signOut(auth);
        toast.info("Logged out successfully!");
    };

    return (
        <div className="login-container">
            <ToastContainer position="top-right" autoClose={2500} />
            <div className="login-box">
                {!user ? (
                    <>
                        <h2>{isSignup ? "Create Account" : "Welcome Back"}</h2>

                        <input
                            type="email"
                            placeholder="Enter your email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                        />

                        <input
                            type="password"
                            placeholder="Enter your password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                        />

                        <button className="login-btn" onClick={handleAuth}>
                            {isSignup ? "Sign Up" : "Login"}
                        </button>

                        <p className="switch-text">
                            {isSignup ? "Already have an account?" : "Don't have an account?"}{" "}
                            <span onClick={() => setIsSignup(!isSignup)}>
                                {isSignup ? "Login" : "Sign Up"}
                            </span>
                        </p>

                        <div className="divider">or</div>

                        <button className="google-btn" onClick={handleGoogleLogin}>
                            <img
                                src="https://cdn-icons-png.flaticon.com/512/300/300221.png"
                                alt="Google"
                            />
                            Sign in with Google
                        </button>
                    </>
                ) : (
                    <div className="logged-in">
                        <h3>Welcome, {user.displayName || user.email}</h3>
                        <button onClick={handleLogout} className="logout-btn">
                            Logout
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Login;
