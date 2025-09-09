import { useEffect, useRef, useState } from 'react';
import wsService, { type EventHandler } from '../services/websocket';

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<Error | null>(null);

  useEffect(() => {
    const connect = async () => {
      try {
        await wsService.connect();
        setIsConnected(true);
        setConnectionError(null);
      } catch (error) {
        setConnectionError(error as Error);
        setIsConnected(false);
      }
    };

    connect();

    return () => {
      wsService.disconnect();
    };
  }, []);

  useEffect(() => {
    const checkConnection = () => {
      setIsConnected(wsService.isConnected);
    };

    const interval = setInterval(checkConnection, 1000);
    return () => clearInterval(interval);
  }, []);

  return { isConnected, connectionError };
}

export function useWebSocketEvent<T = any>(
  eventType: string,
  handler: EventHandler<T>
) {
  const handlerRef = useRef(handler);
  handlerRef.current = handler;

  useEffect(() => {
    const wrappedHandler = (data: T) => handlerRef.current(data);
    
    wsService.on(eventType, wrappedHandler);
    
    return () => {
      wsService.off(eventType, wrappedHandler);
    };
  }, [eventType]);
}