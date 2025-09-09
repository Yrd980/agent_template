import { useState } from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle, Loader2 } from 'lucide-react';
import { useTasks } from '../hooks/useApi';
import { useWebSocketEvent } from '../hooks/useWebSocket';

const statusIcons = {
  pending: Clock,
  running: Loader2,
  completed: CheckCircle,
  failed: XCircle,
  cancelled: AlertCircle,
};

const statusColors = {
  pending: 'text-gray-500 bg-gray-100',
  running: 'text-blue-500 bg-blue-100',
  completed: 'text-green-500 bg-green-100',
  failed: 'text-red-500 bg-red-100',
  cancelled: 'text-yellow-500 bg-yellow-100',
};

export default function TaskMonitor() {
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const { data: tasks = [], refetch, isLoading } = useTasks(selectedStatus === 'all' ? undefined : selectedStatus);

  // WebSocket event handlers
  useWebSocketEvent('task_update', () => {
    refetch();
  });

  const getStatusCounts = () => {
    const counts: Record<string, number> = {
      all: tasks.length,
      pending: 0,
      running: 0,
      completed: 0,
      failed: 0,
      cancelled: 0,
    };

    tasks.forEach(task => {
      counts[task.status]++;
    });

    return counts;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getTimeDuration = (start: string, end?: string) => {
    const startTime = new Date(start);
    const endTime = end ? new Date(end) : new Date();
    const diffMs = endTime.getTime() - startTime.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);

    if (diffSecs < 60) {
      return `${diffSecs}s`;
    } else if (diffMins < 60) {
      return `${diffMins}m ${diffSecs % 60}s`;
    } else {
      return `${diffHours}h ${diffMins % 60}m`;
    }
  };

  const statusCounts = getStatusCounts();

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="bg-gray-200 rounded-lg h-16"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Task Monitor</h2>
        <p className="text-gray-600">
          Monitor real-time task execution and system activities.
        </p>
      </div>

      {/* Status Filter Tabs */}
      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'all', label: 'All Tasks' },
              { id: 'pending', label: 'Pending' },
              { id: 'running', label: 'Running' },
              { id: 'completed', label: 'Completed' },
              { id: 'failed', label: 'Failed' },
              { id: 'cancelled', label: 'Cancelled' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setSelectedStatus(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  selectedStatus === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
                <span className="ml-2 bg-gray-100 text-gray-900 py-0.5 px-2 rounded-full text-xs">
                  {statusCounts[tab.id]}
                </span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tasks List */}
      {tasks.length === 0 ? (
        <div className="text-center py-20">
          <div className="text-6xl mb-4">⚡</div>
          <h3 className="text-xl font-medium text-gray-900 mb-2">No tasks found</h3>
          <p className="text-gray-500">
            {selectedStatus === 'all' 
              ? 'No tasks have been created yet.'
              : `No ${selectedStatus} tasks found.`
            }
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {tasks.map((task) => {
            const StatusIcon = statusIcons[task.status];
            const isRunning = task.status === 'running';
            
            return (
              <div
                key={task.id}
                className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4 flex-1">
                    <div className={`p-2 rounded-lg ${statusColors[task.status]}`}>
                      <StatusIcon className={`w-5 h-5 ${isRunning ? 'animate-spin' : ''}`} />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-2">
                        <h3 className="font-semibold text-gray-900 truncate">
                          {task.description}
                        </h3>
                        <span className="text-xs text-gray-500">
                          Priority: {task.priority}
                        </span>
                      </div>
                      
                      <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                        <span>ID: {task.id.slice(0, 8)}...</span>
                        <span>Type: {task.type}</span>
                        <span>
                          Duration: {getTimeDuration(task.created_at, task.updated_at)}
                        </span>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs text-gray-500">
                        <div>Created: {formatTimestamp(task.created_at)}</div>
                        <div>Updated: {formatTimestamp(task.updated_at)}</div>
                      </div>

                      {/* Task Result */}
                      {task.status === 'completed' && task.result && (
                        <div className="mt-3 p-3 bg-green-50 rounded border">
                          <h4 className="text-sm font-medium text-green-800 mb-1">Result</h4>
                          <pre className="text-xs text-green-700 whitespace-pre-wrap max-h-20 overflow-y-auto">
                            {typeof task.result === 'string' 
                              ? task.result 
                              : JSON.stringify(task.result, null, 2)
                            }
                          </pre>
                        </div>
                      )}

                      {/* Task Error */}
                      {task.status === 'failed' && task.error && (
                        <div className="mt-3 p-3 bg-red-50 rounded border">
                          <h4 className="text-sm font-medium text-red-800 mb-1">Error</h4>
                          <pre className="text-xs text-red-700 whitespace-pre-wrap max-h-20 overflow-y-auto">
                            {task.error}
                          </pre>
                        </div>
                      )}

                      {/* Task Metadata */}
                      {Object.keys(task.metadata).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <h4 className="text-xs font-medium text-gray-500 mb-2">METADATA</h4>
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(task.metadata).map(([key, value]) => (
                              <span
                                key={key}
                                className="inline-flex items-center px-2 py-1 rounded text-xs bg-gray-100 text-gray-700"
                              >
                                {key}: {String(value)}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Task Actions */}
                  <div className="flex items-center space-x-2">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium capitalize ${statusColors[task.status]}`}>
                      {task.status}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Summary Card */}
      <div className="mt-8 bg-gray-50 rounded-lg p-6">
        <h4 className="font-medium text-gray-900 mb-4">Task Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {Object.entries(statusCounts).filter(([key]) => key !== 'all').map(([status, count]) => (
            <div key={status} className="text-center">
              <div className="text-2xl font-bold text-gray-900">{count}</div>
              <div className="text-sm text-gray-600 capitalize">{status}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}