import { Trash2, MessageCircle, Calendar, User } from 'lucide-react';
import { useSessions, useDeleteSession } from '../hooks/useApi';

interface SessionManagerProps {
  currentSessionId: string | null;
  onSessionSelect: (sessionId: string) => void;
}

export default function SessionManager({ currentSessionId, onSessionSelect }: SessionManagerProps) {
  const { data: sessions = [], refetch, isLoading } = useSessions();
  const deleteSessionMutation = useDeleteSession();

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (window.confirm('Are you sure you want to delete this session?')) {
      try {
        await deleteSessionMutation.mutateAsync(sessionId);
        refetch();
        
        // If deleting current session, clear selection
        if (currentSessionId === sessionId) {
          onSessionSelect('');
        }
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getTimeSince = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 60) {
      return `${diffMins} minutes ago`;
    } else if (diffHours < 24) {
      return `${diffHours} hours ago`;
    } else {
      return `${diffDays} days ago`;
    }
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-gray-200 rounded-lg h-20"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Sessions</h2>
        <p className="text-gray-600">
          Manage your conversation sessions. Click on a session to view its messages.
        </p>
      </div>

      {sessions.length === 0 ? (
        <div className="text-center py-20">
          <div className="text-6xl mb-4">💬</div>
          <h3 className="text-xl font-medium text-gray-900 mb-2">No sessions yet</h3>
          <p className="text-gray-500">
            Start a conversation in the Chat tab to create your first session.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => onSessionSelect(session.id)}
              className={`bg-white rounded-lg border-2 p-6 cursor-pointer transition-all hover:shadow-md ${
                currentSessionId === session.id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className={`p-2 rounded-lg ${
                      currentSessionId === session.id ? 'bg-blue-200' : 'bg-gray-100'
                    }`}>
                      <MessageCircle className={`w-5 h-5 ${
                        currentSessionId === session.id ? 'text-blue-600' : 'text-gray-600'
                      }`} />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">
                        Session {session.id.slice(0, 8)}...
                      </h3>
                      <p className="text-sm text-gray-500">
                        {session.message_count} messages
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center space-x-2 text-gray-600">
                      <Calendar className="w-4 h-4" />
                      <span>Created: {formatDate(session.created_at)}</span>
                    </div>
                    <div className="flex items-center space-x-2 text-gray-600">
                      <User className="w-4 h-4" />
                      <span>Last activity: {getTimeSince(session.updated_at)}</span>
                    </div>
                  </div>

                  {/* Session metadata */}
                  {Object.keys(session.metadata).length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <h4 className="text-xs font-medium text-gray-500 mb-2">METADATA</h4>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(session.metadata).map(([key, value]) => (
                          <span
                            key={key}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-gray-100 text-gray-800"
                          >
                            {key}: {String(value)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={(e) => handleDeleteSession(session.id, e)}
                  disabled={deleteSessionMutation.isPending}
                  className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                  title="Delete session"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>

              {currentSessionId === session.id && (
                <div className="mt-4 p-3 bg-blue-100 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-sm font-medium text-blue-800">
                      Currently active session
                    </span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900 mb-2">Session Management</h4>
        <p className="text-sm text-gray-600">
          Sessions automatically persist your conversations. Each session maintains its own 
          context and message history. You can switch between sessions at any time.
        </p>
      </div>
    </div>
  );
}