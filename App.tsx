
import React from 'react';
import { HashRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import AttendanceLogPage from './pages/AttendanceLogPage';
import ManageStudentsPage from './pages/ManageStudentsPage';
import ManageCamerasPage from './pages/ManageCamerasPage';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

const App: React.FC = () => {
  return (
    <AuthProvider>
      <HashRouter>
        <AppContent />
      </HashRouter>
    </AuthProvider>
  );
};

const AppContent: React.FC = () => {
  const { token } = useAuth();
  const location = useLocation();

  if (!token) {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<Navigate to="/login" />} />
      </Routes>
    );
  }

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-background p-6">
          <Routes>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/attendance" element={<AttendanceLogPage />} />
            <Route path="/students" element={<ManageStudentsPage />} />
            <Route path="/cameras" element={<ManageCamerasPage />} />
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </main>
      </div>
    </div>
  );
};

export default App;
