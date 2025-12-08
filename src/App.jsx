import React from "react";
import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import MVP from "./pages/MVP";
import SavedResults from "./pages/SavedResults";
import Login from "./pages/Login";

const App = () => {
  return (
    <>
      <Navbar />
      <div className="page-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/mvp" element={<MVP />} />
          <Route path="/saved" element={<SavedResults />} />
          <Route path="/login" element={<Login />} />
        </Routes>
      </div>
    </>
  );
};

export default App;
