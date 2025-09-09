import { Activity, Users, MessageSquare, CheckCircle, XCircle, Wifi, Server, Cpu, HardDrive } from 'lucide-react';
import { useSystemStats, useHealth } from '../hooks/useApi';

export default function SystemStats() {
  const { data: stats, isLoading: statsLoading } = useSystemStats();
  const { data: health, isLoading: healthLoading } = useHealth();

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else {
      return `${minutes}m`;
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getProgressColor = (percentage: number) => {
    if (percentage < 50) return 'bg-green-500';
    if (percentage < 80) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (statsLoading || healthLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-24"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">System Statistics</h2>
        <p className="text-gray-600">
          Real-time system health and performance metrics.
        </p>
      </div>

      {/* Health Status */}
      <div className="mb-6">
        <div className={`inline-flex items-center px-4 py-2 rounded-full ${
          health?.status === 'healthy' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${
            health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
          }`} />
          System Status: {health?.status || 'Unknown'}
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {/* Uptime */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Uptime</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats ? formatUptime(stats.uptime) : '0m'}
              </p>
            </div>
            <div className="p-3 bg-blue-100 rounded-lg">
              <Server className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </div>

        {/* Active Sessions */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Active Sessions</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.active_sessions || 0}
              </p>
            </div>
            <div className="p-3 bg-green-100 rounded-lg">
              <Users className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </div>

        {/* Total Messages */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Messages</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.total_messages || 0}
              </p>
            </div>
            <div className="p-3 bg-purple-100 rounded-lg">
              <MessageSquare className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </div>

        {/* WebSocket Connections */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">WS Connections</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats?.websocket_connections || 0}
              </p>
            </div>
            <div className="p-3 bg-indigo-100 rounded-lg">
              <Wifi className="w-6 h-6 text-indigo-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Task Statistics */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Task Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Active Tasks</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats?.active_tasks || 0}
                </p>
              </div>
              <div className="p-3 bg-yellow-100 rounded-lg">
                <Activity className="w-6 h-6 text-yellow-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Completed Tasks</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats?.completed_tasks || 0}
                </p>
              </div>
              <div className="p-3 bg-green-100 rounded-lg">
                <CheckCircle className="w-6 h-6 text-green-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Failed Tasks</p>
                <p className="text-2xl font-bold text-gray-900">
                  {stats?.failed_tasks || 0}
                </p>
              </div>
              <div className="p-3 bg-red-100 rounded-lg">
                <XCircle className="w-6 h-6 text-red-600" />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Resources */}
      <div className="mb-8">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Resources</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* CPU Usage */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Cpu className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-900">CPU Usage</span>
              </div>
              <span className="text-sm font-medium text-gray-600">
                {stats?.cpu_usage?.toFixed(1) || 0}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-300 ${
                  getProgressColor(stats?.cpu_usage || 0)
                }`}
                style={{ width: `${Math.min(stats?.cpu_usage || 0, 100)}%` }}
              />
            </div>
          </div>

          {/* Memory Usage */}
          <div className="bg-white rounded-lg border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <HardDrive className="w-5 h-5 text-gray-600" />
                <span className="font-medium text-gray-900">Memory Usage</span>
              </div>
              <span className="text-sm font-medium text-gray-600">
                {formatBytes(stats?.memory_usage || 0)}
              </span>
            </div>
            <div className="text-xs text-gray-500 mb-2">
              {/* Assuming a rough percentage calculation */}
              {((stats?.memory_usage || 0) / (1024 * 1024 * 1024) * 100).toFixed(1)}% of available memory
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all duration-300 ${
                  getProgressColor((stats?.memory_usage || 0) / (1024 * 1024 * 1024) * 100)
                }`}
                style={{ width: `${Math.min((stats?.memory_usage || 0) / (1024 * 1024 * 1024) * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Last Updated */}
      <div className="text-sm text-gray-500 text-center">
        Last updated: {new Date().toLocaleString()}
        <br />
        Metrics refresh every 5 seconds
      </div>
    </div>
  );
}