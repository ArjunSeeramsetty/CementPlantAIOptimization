# src/cement_ai_platform/config/otel_tracer.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import os
import logging

logger = logging.getLogger(__name__)

def setup_tracing(service_name: str, project_id: str = None):
    """
    Setup OpenTelemetry tracing with Cloud Trace export.
    
    Args:
        service_name (str): Name of the service for tracing
        project_id (str, optional): GCP project ID. If None, uses environment variable.
        
    Returns:
        trace.Tracer: Configured tracer instance
    """
    try:
        # Get project ID
        if project_id is None:
            project_id = os.getenv('CEMENT_GCP_PROJECT', 'cement-ai-opt-38517')
        
        # Create resource with service information
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "service.namespace": "cement-ai-platform"
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Try to create Cloud Trace exporter, fallback to console if it fails
        try:
            from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
            exporter = CloudTraceSpanExporter(project_id=project_id)
            logger.info("Using Cloud Trace exporter")
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            exporter = ConsoleSpanExporter()
            logger.warning("Cloud Trace exporter not available, using console exporter")
        
        # Create span processor
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)
        
        # Set the global tracer provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        tracer = trace.get_tracer(service_name)
        
        logger.info(f"OpenTelemetry tracing initialized for service: {service_name}")
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
        # Return a no-op tracer if setup fails
        return trace.NoOpTracer()

def get_tracer(service_name: str = "cement-ai-platform") -> trace.Tracer:
    """
    Get a tracer instance for the service.
    
    Args:
        service_name (str): Name of the service
        
    Returns:
        trace.Tracer: Tracer instance
    """
    return trace.get_tracer(service_name)

def trace_function(func_name: str = None):
    """
    Decorator to trace function execution.
    
    Args:
        func_name (str, optional): Custom name for the span
        
    Returns:
        Callable: Decorated function with tracing
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            span_name = func_name or f"{func.__module__}.{func.__name__}"
            
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function arguments as span attributes
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.module", func.__module__)
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Mark span as successful
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Mark span as failed
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.set_attribute("error.message", str(e))
                    raise
        
        return wrapper
    return decorator

def trace_gcp_operation(operation_name: str):
    """
    Decorator specifically for tracing GCP operations.
    
    Args:
        operation_name (str): Name of the GCP operation
        
    Returns:
        Callable: Decorated function with GCP-specific tracing
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            with tracer.start_as_current_span(f"gcp.{operation_name}") as span:
                try:
                    # Add GCP-specific attributes
                    span.set_attribute("gcp.operation", operation_name)
                    span.set_attribute("gcp.service", func.__module__.split('.')[-1])
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Mark span as successful
                    span.set_status(trace.Status(trace.StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    # Mark span as failed
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    span.set_attribute("error.message", str(e))
                    span.set_attribute("error.type", type(e).__name__)
                    raise
        
        return wrapper
    return decorator
