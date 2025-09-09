import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useWebSocket } from './hooks/useWebSocket';
import ChatInterface from './components/ChatInterface';
import SessionManager from './components/SessionManager';
import TaskMonitor from './components/TaskMonitor';
import SystemStats from './components/SystemStats';
import { Activity, MessageCircle, Settings, BarChart3 } from 'lucide-react';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

type TabType = 'chat' | 'sessions' | 'tasks' | 'stats';

function AppContent() {
  const [activeTab, setActiveTab] = useState<TabType>('chat');
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const { isConnected, connectionError } = useWebSocket();

  const tabs = [
    { id: 'chat', label: 'Chat', icon: MessageCircle },
    { id: 'sessions', label: 'Sessions', icon: Settings },
    { id: 'tasks', label: 'Tasks', icon: Activity },
    { id: 'stats', label: 'Stats', icon: BarChart3 },
  ] as const;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-gray-900">Agent Template</h1>
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
              isConnected 
                ? 'bg-green-100 text-green-800' 
                : 'bg-red-100 text-red-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
          
          {connectionError && (
            <div className="text-red-600 text-sm">
              Connection Error: {connectionError.message}
            </div>
          )}
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="px-6">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as TabType)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {activeTab === 'chat' && (
          <ChatInterface 
            sessionId={currentSessionId}
            onSessionChange={setCurrentSessionId}
          />
        )}
        {activeTab === 'sessions' && (
          <SessionManager 
            currentSessionId={currentSessionId}
            onSessionSelect={setCurrentSessionId}
          />
        )}
        {activeTab === 'tasks' && <TaskMonitor />}
        {activeTab === 'stats' && <SystemStats />}
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  );
}

export default App
