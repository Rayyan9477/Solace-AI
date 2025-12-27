"""
Supervision Dashboard and Reporting Interface.

This module provides a comprehensive web-based dashboard for monitoring
and managing the SupervisorAgent system, including real-time metrics,
quality reports, and compliance monitoring.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional
import json
import asyncio
import base64
from io import BytesIO

# Dashboard configuration
API_BASE_URL = "http://localhost:8000/api"
REFRESH_INTERVAL = 30  # seconds
REQUEST_TIMEOUT = 10  # seconds - timeout for all HTTP requests to prevent hanging

def main():
    """Main dashboard application."""
    st.set_page_config(
        page_title="SupervisorAgent Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .critical-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .warning-alert {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #4caf50;
        font-weight: bold;
    }
    .warning-metric {
        color: #ff9800;
        font-weight: bold;
    }
    .critical-metric {
        color: #f44336;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🛡️ SupervisorAgent Dashboard")
    
    # Check API connectivity
    api_status = check_api_connectivity()
    if not api_status:
        st.error("❌ Cannot connect to SupervisorAgent API. Please ensure the API server is running.")
        st.stop()
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🏠 Overview",
            "📊 Real-time Monitoring",
            "👥 Agent Quality Reports",
            "🔍 Session Analysis",
            "📋 Compliance Reports",
            "⚙️ Configuration",
            "🚨 Alerts & Notifications"
        ]
    )
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview_page()
    elif page == "📊 Real-time Monitoring":
        show_monitoring_page()
    elif page == "👥 Agent Quality Reports":
        show_agent_quality_page()
    elif page == "🔍 Session Analysis":
        show_session_analysis_page()
    elif page == "📋 Compliance Reports":
        show_compliance_page()
    elif page == "⚙️ Configuration":
        show_configuration_page()
    elif page == "🚨 Alerts & Notifications":
        show_alerts_page()

def check_api_connectivity() -> bool:
    """Check if the API server is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/supervision/status", timeout=5)
        return response.status_code == 200
    except (requests.RequestException, ConnectionError, TimeoutError):
        return False

def show_overview_page():
    """Display the overview dashboard page."""
    st.title("🏠 SupervisorAgent Overview")
    st.markdown("---")
    
    # Get supervision status
    status_data = get_supervision_status()
    if not status_data:
        st.error("Failed to load supervision status")
        return
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if status_data.get("supervision_enabled") else "🔴"
        st.metric(
            "Supervision Status",
            f"{status_color} {'Enabled' if status_data.get('supervision_enabled') else 'Disabled'}"
        )
    
    with col2:
        st.metric("Active Workflows", status_data.get("active_workflows", 0))
    
    with col3:
        st.metric("Total Agents", status_data.get("total_agents", 0))
    
    with col4:
        supervisor_status = "🟢 Active" if status_data.get("supervisor_agent_active") else "🔴 Inactive"
        st.metric("Supervisor Agent", supervisor_status)
    
    st.markdown("---")
    
    # Get supervision summary
    summary_data = get_supervision_summary()
    if not summary_data:
        st.warning("Supervision summary not available")
        return
    
    # Real-time metrics overview
    st.subheader("📈 Real-time Metrics")
    
    metrics = summary_data.get("real_time_metrics", {})
    validation_perf = metrics.get("validation_performance", {})
    
    if validation_perf:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = validation_perf.get("accuracy", {}).get("mean", 0)
            st.metric(
                "Average Accuracy",
                f"{accuracy:.2%}" if accuracy else "N/A",
                delta=None
            )
        
        with col2:
            processing_time = validation_perf.get("processing_time", {}).get("mean", 0)
            st.metric(
                "Avg Processing Time",
                f"{processing_time:.2f}s" if processing_time else "N/A"
            )
        
        with col3:
            blocked_rate = validation_perf.get("blocked_response_rate", 0)
            st.metric(
                "Blocked Response Rate",
                f"{blocked_rate:.1%}" if blocked_rate else "N/A"
            )
    
    # Recent alerts
    st.subheader("🚨 Recent Alerts")
    alerts_data = get_active_alerts()
    
    if alerts_data and alerts_data.get("count", 0) > 0:
        for alert in alerts_data["alerts"][:5]:  # Show top 5 alerts
            alert_class = "critical-alert" if alert["level"] == "critical" else "warning-alert"
            st.markdown(
                f'<div class="{alert_class}"><strong>{alert["title"]}</strong><br>'
                f'{alert["description"]}<br>'
                f'<small>Time: {alert["timestamp"]}</small></div>',
                unsafe_allow_html=True
            )
    else:
        st.success("✅ No active alerts")

def show_monitoring_page():
    """Display the real-time monitoring page."""
    st.title("📊 Real-time Monitoring")
    st.markdown("---")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        st.rerun()
    
    # Time window selector
    time_window = st.sidebar.selectbox(
        "Time Window",
        ["1 hour", "6 hours", "24 hours", "7 days"],
        index=2
    )
    
    time_window_hours = {
        "1 hour": 1,
        "6 hours": 6,
        "24 hours": 24,
        "7 days": 168
    }[time_window]
    
    # Get monitoring data
    summary_data = get_supervision_summary(time_window_hours)
    if not summary_data:
        st.error("Failed to load monitoring data")
        return
    
    # Key performance indicators
    st.subheader("📊 Key Performance Indicators")
    
    metrics = summary_data.get("real_time_metrics", {})
    validation_perf = metrics.get("validation_performance", {})
    system_health = metrics.get("system_health", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = validation_perf.get("accuracy", {}).get("mean", 0)
        accuracy_color = get_metric_color(accuracy, 0.8, 0.6)
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 class="{accuracy_color}">Validation Accuracy</h3>'
            f'<h2>{accuracy:.1%}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        blocked_rate = validation_perf.get("blocked_response_rate", 0)
        blocked_color = get_metric_color(1 - blocked_rate, 0.9, 0.75)  # Inverted: lower is better
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 class="{blocked_color}">Blocked Rate</h3>'
            f'<h2>{blocked_rate:.1%}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        processing_time = validation_perf.get("processing_time", {}).get("mean", 0)
        time_color = get_metric_color(1 / max(processing_time, 0.1), 0.5, 0.2)  # Inverted: lower is better
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 class="{time_color}">Processing Time</h3>'
            f'<h2>{processing_time:.2f}s</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        cpu_usage = system_health.get("cpu_usage", {}).get("mean", 0)
        cpu_color = get_metric_color(1 - cpu_usage / 100, 0.8, 0.6)  # Inverted: lower is better
        st.markdown(
            f'<div class="metric-card">'
            f'<h3 class="{cpu_color}">CPU Usage</h3>'
            f'<h2>{cpu_usage:.1f}%</h2>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Performance trends chart
    st.subheader("📈 Performance Trends")
    
    # Generate sample trend data (in real implementation, this would come from the API)
    trend_data = generate_sample_trend_data(time_window_hours)
    
    if trend_data:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data["timestamp"],
            y=trend_data["accuracy"],
            mode='lines+markers',
            name='Validation Accuracy',
            line=dict(color='#2E8B57')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data["timestamp"],
            y=trend_data["processing_time"],
            mode='lines+markers',
            name='Processing Time (s)',
            yaxis='y2',
            line=dict(color='#FF6B6B')
        ))
        
        fig.update_layout(
            title="Performance Trends Over Time",
            xaxis_title="Time",
            yaxis=dict(title="Accuracy (%)", side="left"),
            yaxis2=dict(title="Processing Time (s)", side="right", overlaying="y"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_agent_quality_page():
    """Display the agent quality reports page."""
    st.title("👥 Agent Quality Reports")
    st.markdown("---")
    
    # Agent selector
    available_agents = get_available_agents()
    selected_agent = st.selectbox(
        "Select Agent",
        ["All Agents"] + available_agents,
        index=0
    )
    
    # Get quality report
    if selected_agent == "All Agents":
        report_data = get_agent_quality_report(None)
    else:
        report_data = get_agent_quality_report(selected_agent)
    
    if not report_data:
        st.error("Failed to load quality report")
        return
    
    # Quality summary
    st.subheader("📊 Quality Summary")
    
    perf_summary = report_data.get("performance_summary", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = perf_summary.get("average_accuracy", 0)
        st.metric("Average Accuracy", f"{accuracy:.1%}")
    
    with col2:
        consistency = perf_summary.get("average_consistency", 0)
        st.metric("Consistency Score", f"{consistency:.1%}")
    
    with col3:
        validations = perf_summary.get("total_validations", 0)
        st.metric("Total Validations", validations)
    
    with col4:
        trend = perf_summary.get("performance_trend", "stable")
        trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}
        st.metric("Trend", f"{trend_emoji.get(trend, '➡️')} {trend.title()}")
    
    # Quality indicators
    st.subheader("🎯 Quality Indicators")
    
    quality_indicators = report_data.get("quality_indicators", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        blocked_responses = quality_indicators.get("blocked_responses", 0)
        critical_issues = quality_indicators.get("critical_issues", 0)
        
        st.markdown("**🚫 Blocked Responses**")
        st.metric("Count", blocked_responses)
        
        st.markdown("**⚠️ Critical Issues**")
        st.metric("Count", critical_issues)
    
    with col2:
        user_satisfaction = quality_indicators.get("user_satisfaction", 0)
        
        st.markdown("**😊 User Satisfaction**")
        st.metric("Average Rating", f"{user_satisfaction:.1f}/5.0")
        
        # Satisfaction gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=user_satisfaction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "User Satisfaction"},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 2.5], 'color': "lightgray"},
                    {'range': [2.5, 4], 'color': "yellow"},
                    {'range': [4, 5], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top issues and recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Top Issues")
        top_issues = report_data.get("top_issues", [])
        
        if top_issues:
            for i, issue in enumerate(top_issues[:5], 1):
                st.write(f"{i}. {issue}")
        else:
            st.success("No significant issues identified")
    
    with col2:
        st.subheader("💡 Recommendations")
        recommendations = report_data.get("recommendations", [])
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("No specific recommendations at this time")

def show_session_analysis_page():
    """Display the session analysis page."""
    st.title("🔍 Session Analysis")
    st.markdown("---")
    
    # Session ID input
    session_id = st.text_input("Enter Session ID", placeholder="session_123456789")
    
    if not session_id:
        st.info("Please enter a session ID to analyze")
        return
    
    # Analyze button
    if st.button("🔍 Analyze Session"):
        analysis_data = get_session_analysis(session_id)
        
        if not analysis_data:
            st.error("Failed to load session analysis or session not found")
            return
        
        # Session overview
        st.subheader("📋 Session Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session ID", session_id)
        
        with col2:
            audit_events = analysis_data.get("audit_events_count", 0)
            st.metric("Audit Events", audit_events)
        
        with col3:
            critical_issues = analysis_data.get("critical_issues", 0)
            status_color = "🔴" if critical_issues > 0 else "🟢"
            st.metric("Critical Issues", f"{status_color} {critical_issues}")
        
        # Event summary
        st.subheader("📊 Event Summary")
        
        event_summary = analysis_data.get("event_summary", {})
        
        if event_summary:
            # Create pie chart of event types
            fig = px.pie(
                values=list(event_summary.values()),
                names=list(event_summary.keys()),
                title="Distribution of Event Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events found for this session")
        
        # Critical details
        critical_details = analysis_data.get("critical_details", [])
        
        if critical_details:
            st.subheader("⚠️ Critical Issues Details")
            
            for detail in critical_details:
                st.markdown(
                    f'<div class="critical-alert">'
                    f'<strong>{detail["event_type"].replace("_", " ").title()}</strong> '
                    f'- {detail["severity"].upper()}<br>'
                    f'{detail["description"]}<br>'
                    f'<small>Time: {detail["timestamp"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        
        # Supervisor summary
        supervisor_summary = analysis_data.get("supervisor_summary")
        
        if supervisor_summary:
            st.subheader("🛡️ Supervisor Analysis")
            
            quality_metrics = supervisor_summary.get("quality_metrics", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = quality_metrics.get("average_accuracy", 0)
                st.metric("Average Accuracy", f"{accuracy:.1%}")
            
            with col2:
                consistency = quality_metrics.get("average_consistency", 0)
                st.metric("Consistency", f"{consistency:.1%}")
            
            with col3:
                pass_rate = quality_metrics.get("pass_rate", 0)
                st.metric("Pass Rate", f"{pass_rate:.1%}")

def show_compliance_page():
    """Display the compliance reports page."""
    st.title("📋 Compliance Reports")
    st.markdown("---")
    
    # Compliance standard selector
    compliance_standards = ["HIPAA", "GDPR", "SOC2", "Clinical_Trials", "FDA_Software"]
    selected_standard = st.selectbox("Select Compliance Standard", compliance_standards)
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", value=date.today())
    
    # Generate report button
    if st.button("📋 Generate Compliance Report"):
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
        
        # Show loading spinner
        with st.spinner("Generating compliance report..."):
            report_data = generate_compliance_report(
                selected_standard.lower(),
                start_date.isoformat(),
                end_date.isoformat()
            )
        
        if not report_data:
            st.error("Failed to generate compliance report")
            return
        
        # Display compliance report
        compliance_report = report_data.get("compliance_report", {})
        
        # Report header
        st.subheader(f"📋 {selected_standard} Compliance Report")
        st.markdown(f"**Period:** {start_date} to {end_date}")
        st.markdown(f"**Generated:** {report_data.get('generated_timestamp', 'N/A')}")
        
        # Compliance score
        compliance_score = compliance_report.get("compliance_score", 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score_color = get_compliance_color(compliance_score)
            st.markdown(
                f'<div class="metric-card">'
                f'<h3 class="{score_color}">Compliance Score</h3>'
                f'<h2>{compliance_score:.1%}</h2>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            total_events = compliance_report.get("total_events", 0)
            st.metric("Total Events", total_events)
        
        with col3:
            violations = compliance_report.get("violations_found", 0)
            st.metric("Violations Found", violations)
        
        # Critical findings
        critical_findings = compliance_report.get("critical_findings", [])
        
        if critical_findings:
            st.subheader("⚠️ Critical Findings")
            for finding in critical_findings:
                st.markdown(
                    f'<div class="critical-alert">{finding}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("✅ No critical findings")
        
        # Recommendations
        recommendations = compliance_report.get("recommendations", [])
        
        if recommendations:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        
        # Download report
        if st.button("💾 Download Report"):
            # Create downloadable report
            report_json = json.dumps(report_data, indent=2)
            b64 = base64.b64encode(report_json.encode()).decode()
            
            href = f'<a href="data:application/json;base64,{b64}" download="compliance_report_{selected_standard}_{start_date}_{end_date}.json">Download JSON Report</a>'
            st.markdown(href, unsafe_allow_html=True)

def show_configuration_page():
    """Display the configuration management page."""
    st.title("⚙️ Configuration Management")
    st.markdown("---")
    
    st.info("🚧 Configuration management interface is under development. Use API endpoints for now.")
    
    # Show current configuration status
    status_data = get_supervision_status()
    
    if status_data:
        st.subheader("🔧 Current Configuration")
        
        config_data = {
            "Supervision Enabled": status_data.get("supervision_enabled", False),
            "Supervisor Agent Active": status_data.get("supervisor_agent_active", False),
            "Metrics Collection Active": status_data.get("metrics_collector_active", False),
            "Audit Trail Active": status_data.get("audit_trail_active", False),
            "Total Agents": status_data.get("total_agents", 0),
            "Active Workflows": status_data.get("active_workflows", 0)
        }
        
        for key, value in config_data.items():
            if isinstance(value, bool):
                status_emoji = "✅" if value else "❌"
                st.write(f"**{key}:** {status_emoji} {value}")
            else:
                st.write(f"**{key}:** {value}")

def show_alerts_page():
    """Display the alerts and notifications page."""
    st.title("🚨 Alerts & Notifications")
    st.markdown("---")
    
    # Get active alerts
    alerts_data = get_active_alerts()
    
    if not alerts_data:
        st.error("Failed to load alerts")
        return
    
    alert_count = alerts_data.get("count", 0)
    
    # Alert summary
    st.subheader(f"📊 Alert Summary ({alert_count} active)")
    
    if alert_count == 0:
        st.success("✅ No active alerts")
        return
    
    # Group alerts by level
    alerts = alerts_data.get("alerts", [])
    alert_levels = {}
    
    for alert in alerts:
        level = alert["level"]
        if level not in alert_levels:
            alert_levels[level] = []
        alert_levels[level].append(alert)
    
    # Display alerts by level
    for level in ["critical", "warning", "info"]:
        if level in alert_levels:
            level_alerts = alert_levels[level]
            level_emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}
            
            st.subheader(f"{level_emoji[level]} {level.title()} Alerts ({len(level_alerts)})")
            
            for alert in level_alerts:
                alert_class = "critical-alert" if level == "critical" else "warning-alert"
                
                # Create alert card with resolve button
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(
                        f'<div class="{alert_class}">'
                        f'<strong>{alert["title"]}</strong><br>'
                        f'{alert["description"]}<br>'
                        f'<small>Metric: {alert["metric_name"]} | '
                        f'Value: {alert["current_value"]} | '
                        f'Threshold: {alert["threshold_value"]}<br>'
                        f'Time: {alert["timestamp"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col2:
                    if st.button(f"Resolve", key=f"resolve_{alert['alert_id']}"):
                        if resolve_alert(alert["alert_id"]):
                            st.success("Alert resolved!")
                            st.rerun()
                        else:
                            st.error("Failed to resolve alert")

# Utility functions
def get_supervision_status() -> Optional[Dict]:
    """Get supervision system status."""
    try:
        response = requests.get(f"{API_BASE_URL}/supervision/status", timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def get_supervision_summary(time_window_hours: int = 24) -> Optional[Dict]:
    """Get supervision summary."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/supervision/summary",
            params={"time_window_hours": time_window_hours},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def get_agent_quality_report(agent_name: Optional[str]) -> Optional[Dict]:
    """Get agent quality report."""
    try:
        if agent_name:
            response = requests.get(
                f"{API_BASE_URL}/supervision/agent-quality/{agent_name}",
                timeout=REQUEST_TIMEOUT
            )
        else:
            # Get report for all agents
            response = requests.get(
                f"{API_BASE_URL}/supervision/agent-quality/all",
                timeout=REQUEST_TIMEOUT
            )

        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def get_session_analysis(session_id: str) -> Optional[Dict]:
    """Get session analysis."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/supervision/session-analysis/{session_id}",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def generate_compliance_report(standard: str, start_date: str, end_date: str) -> Optional[Dict]:
    """Generate compliance report."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/supervision/compliance-report",
            json={
                "compliance_standard": standard,
                "start_date": start_date,
                "end_date": end_date
            },
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def get_active_alerts() -> Optional[Dict]:
    """Get active alerts."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/supervision/alerts",
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
    except (requests.RequestException, ConnectionError, TimeoutError, ValueError):
        pass
    return None

def resolve_alert(alert_id: str) -> bool:
    """Resolve an alert."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/supervision/alerts/{alert_id}/resolve",
            timeout=REQUEST_TIMEOUT
        )
        return response.status_code == 200
    except (requests.RequestException, ConnectionError, TimeoutError):
        return False

def get_available_agents() -> List[str]:
    """Get list of available agents."""
    # In a real implementation, this would call an API endpoint
    return [
        "therapy_agent",
        "emotion_agent",
        "safety_agent",
        "diagnosis_agent",
        "chat_agent"
    ]

def get_metric_color(value: float, good_threshold: float, warning_threshold: float) -> str:
    """Get CSS class for metric color based on thresholds."""
    if value >= good_threshold:
        return "success-metric"
    elif value >= warning_threshold:
        return "warning-metric"
    else:
        return "critical-metric"

def get_compliance_color(score: float) -> str:
    """Get CSS class for compliance score color."""
    if score >= 0.9:
        return "success-metric"
    elif score >= 0.7:
        return "warning-metric"
    else:
        return "critical-metric"

def generate_sample_trend_data(hours: int) -> Dict:
    """Generate sample trend data for visualization."""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    time_points = pd.date_range(start_time, end_time, periods=min(hours, 50))
    
    # Generate sample data with some realistic patterns
    np.random.seed(42)  # For reproducible sample data
    
    base_accuracy = 0.85
    accuracy_noise = np.random.normal(0, 0.05, len(time_points))
    accuracy = np.clip(base_accuracy + accuracy_noise, 0.5, 1.0)
    
    base_processing_time = 1.2
    time_noise = np.random.normal(0, 0.3, len(time_points))
    processing_time = np.clip(base_processing_time + time_noise, 0.5, 5.0)
    
    return {
        "timestamp": time_points.tolist(),
        "accuracy": accuracy.tolist(),
        "processing_time": processing_time.tolist()
    }

if __name__ == "__main__":
    main()