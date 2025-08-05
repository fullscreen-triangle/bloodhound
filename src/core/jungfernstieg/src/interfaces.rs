//! # Jungfernstieg System Interfaces
//!
//! External interfaces for the Jungfernstieg biological-virtual neural symbiosis system.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::{JungfernstigSystem, JungfernstigMetrics, BiologicalNeuralNetworkSpec};

/// REST API interface for Jungfernstieg system
#[derive(Debug)]
pub struct JungfernstigRESTInterface {
    /// Reference to the Jungfernstieg system
    pub system: JungfernstigSystem,
}

/// WebSocket interface for real-time monitoring
#[derive(Debug)]
pub struct JungfernstigWebSocketInterface {
    /// Reference to the Jungfernstieg system
    pub system: JungfernstigSystem,
}

/// Command-line interface
#[derive(Debug)]
pub struct JungfernstigCLI {
    /// Reference to the Jungfernstieg system
    pub system: JungfernstigSystem,
}

/// API request structures
#[derive(Debug, Serialize, Deserialize)]
pub struct AddNetworkRequest {
    pub network_spec: BiologicalNeuralNetworkSpec,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddNetworkResponse {
    pub network_id: Uuid,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetMetricsResponse {
    pub metrics: JungfernstigMetrics,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatusResponse {
    pub system_id: Uuid,
    pub status: String,
    pub uptime: f64,
    pub neural_networks_count: usize,
    pub overall_viability: f64,
}

impl JungfernstigRESTInterface {
    /// Create a new REST interface
    pub fn new(system: JungfernstigSystem) -> Self {
        Self { system }
    }
    
    /// Get system status
    pub async fn get_status(&self) -> Result<SystemStatusResponse> {
        let metrics = self.system.get_metrics().await?;
        let networks = self.system.neural_networks.read().await;
        
        Ok(SystemStatusResponse {
            system_id: self.system.id,
            status: "operational".to_string(),
            uptime: 3600.0, // Placeholder
            neural_networks_count: networks.len(),
            overall_viability: metrics.neural_viability,
        })
    }
    
    /// Get system metrics
    pub async fn get_metrics(&self) -> Result<GetMetricsResponse> {
        let metrics = self.system.get_metrics().await?;
        
        Ok(GetMetricsResponse {
            metrics,
            timestamp: chrono::Utc::now(),
        })
    }
    
    /// Add a biological neural network
    pub async fn add_neural_network(&self, request: AddNetworkRequest) -> Result<AddNetworkResponse> {
        let network_id = self.system.add_biological_neural_network(request.network_spec).await?;
        
        Ok(AddNetworkResponse {
            network_id,
            status: "created".to_string(),
        })
    }
}

impl JungfernstigWebSocketInterface {
    /// Create a new WebSocket interface
    pub fn new(system: JungfernstigSystem) -> Self {
        Self { system }
    }
    
    /// Start real-time monitoring stream
    pub async fn start_monitoring_stream(&self) -> Result<()> {
        // Implementation would stream metrics in real-time
        Ok(())
    }
}

impl JungfernstigCLI {
    /// Create a new CLI interface
    pub fn new(system: JungfernstigSystem) -> Self {
        Self { system }
    }
    
    /// Start the system
    pub async fn start(&self) -> Result<()> {
        self.system.start().await
    }
    
    /// Stop the system
    pub async fn stop(&self) -> Result<()> {
        self.system.shutdown().await
    }
    
    /// Show system status
    pub async fn status(&self) -> Result<()> {
        let metrics = self.system.get_metrics().await?;
        println!("Jungfernstieg System Status:");
        println!("  Neural Viability: {:.1}%", metrics.neural_viability);
        println!("  S-Entropy Efficiency: {:.1}%", metrics.s_entropy_efficiency * 100.0);
        println!("  Virtual Blood Quality: {:.1}%", metrics.virtual_blood_quality * 100.0);
        println!("  System Coherence: {:.1}%", metrics.system_coherence * 100.0);
        Ok(())
    }
}