import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
    apiKey: "AIzaSyC6tm1BW3xLQKyc9xeYIxBzYvTeeCWmeFY",
    authDomain: "brain-mri-segmentation.firebaseapp.com",
    projectId: "brain-mri-segmentation",
    storageBucket: "brain-mri-segmentation.appspot.com", 
    messagingSenderId: "854127885230",
    appId: "1:854127885230:web:eec9c53f613159084f263f",
    measurementId: "G-8Z83405QK7"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
export const googleProvider = new GoogleAuthProvider(); 
export const analytics = getAnalytics(app);
