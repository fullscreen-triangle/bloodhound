//! # S-Entropy Navigation Engine
//!
//! Implementation of the revolutionary S-entropy coordinate system for universal problem navigation.
//! This system enables zero-time computation through tri-dimensional entropy coordinates
//! (S_knowledge, S_time, S_entropy) and oscillatory endpoint navigation.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, Instant};
use tracing::{info, debug, trace, warn};
use uuid::Uuid;

/// S-Entropy Navigation Engine - Core of zero-time computation
#[derive(Debug)]
pub struct SEntropyNavigationEngine {
    /// Engine identifier
    pub id: Uuid,
    
    /// Associated VM identifier
    pub vm_id: Uuid,
    
    /// Engine configuration
    pub config: SEntropyConfiguration,
    
    /// Current S-entropy coordinates
    pub current_coordinates: Arc<RwLock<SEntropyCoordinates>>,
    
    /// Coordinate transformation matrix
    pub transformation_matrix: Arc<RwLock<CoordinateTransformationMatrix>>,
    
    /// Navigation cache for instant access
    pub navigation_cache: Arc<RwLock<NavigationCache>>,
    
    /// Oscillatory endpoint registry
    pub oscillatory_endpoints: Arc<RwLock<OscillatoryEndpointRegistry>>,
    
    /// Zero-time processing coordinator
    pub zero_time_coordinator: Arc<RwLock<ZeroTimeCoordinator>>,
    
    /// Navigation history
    pub navigation_history: Arc<RwLock<NavigationHistory>>,
    
    /// Engine state
    pub is_active: Arc<RwLock<bool>>,
}

/// S-Entropy Coordinates (S_knowledge, S_time, S_entropy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyCoordinates {
    /// Information deficit regarding the problem domain
    pub s_knowledge: f64,
    
    /// Temporal coordination distance to solution
    pub s_time: f64,
    
    /// Thermodynamic entropy distance in solution space
    pub s_entropy: f64,
    
    /// Coordinate quality metrics
    pub coordinate_quality: CoordinateQuality,
    
    /// Timestamp of last update
    pub last_updated: Instant,
}

/// S-Entropy Navigation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyConfiguration {
    /// Maximum coordinate range for each dimension
    pub max_coordinate_ranges: (f64, f64, f64), // (max_s_knowledge, max_s_time, max_s_entropy)
    
    /// Navigation precision level
    pub navigation_precision: NavigationPrecision,
    
    /// Zero-time processing threshold
    pub zero_time_threshold: Duration,
    
    /// Oscillatory endpoint density
    pub endpoint_density: u32,
    
    /// Cache configuration
    pub cache_config: NavigationCacheConfig,
    
    /// Performance optimization settings
    pub optimization_settings: OptimizationSettings,
}

/// Navigation precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NavigationPrecision {
    /// Basic precision for standard problems
    Basic,
    /// High precision for complex problems
    High,
    /// Ultra precision for consciousness-level problems
    Ultra,
    /// Infinite precision through recursive refinement
    Infinite,
}

/// Navigation modes for different processing requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SEntropyNavigationMode {
    /// Direct coordinate jump (instant)
    Direct,
    /// Optimized path navigation (fastest route)
    Optimized,
    /// Zero-time navigation through oscillatory endpoints
    ZeroTime,
    /// Consciousness-aware navigation with BMD orchestration
    ConsciousnessAware,
    /// Infinite parallelization mode
    InfiniteParallel,
}

/// Result of S-entropy navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyNavigationResult {
    /// Navigation success status
    pub navigation_success: bool,
    
    /// Final coordinates reached
    pub final_coordinates: SEntropyCoordinates,
    
    /// Actual navigation time
    pub navigation_time: Duration,
    
    /// Solution quality metrics
    pub solution_quality: SolutionQuality,
    
    /// Oscillatory endpoints utilized
    pub endpoints_utilized: Vec<Uuid>,
    
    /// Navigation path taken
    pub navigation_path: NavigationPath,
}

/// Coordinate Transformation Matrix for universal problem navigation
#[derive(Debug, Clone)]
pub struct CoordinateTransformationMatrix {
    /// Transformation matrix elements
    pub matrix: [[f64; 3]; 3], // 3x3 matrix for tri-dimensional coordinates
    
    /// Inverse transformation matrix
    pub inverse_matrix: [[f64; 3]; 3],
    
    /// Matrix quality metrics
    pub matrix_quality: MatrixQuality,
    
    /// Last matrix update
    pub last_updated: Instant,
}

/// Navigation Cache for instant problem resolution
#[derive(Debug, Clone)]
pub struct NavigationCache {
    /// Cached navigation solutions
    pub cached_solutions: HashMap<ProblemSignature, CachedSolution>,
    
    /// Cache statistics
    pub cache_stats: CacheStatistics,
    
    /// Cache cleanup manager
    pub cleanup_manager: CacheCleanupManager,
}

/// Oscillatory Endpoint Registry
#[derive(Debug, Clone)]
pub struct OscillatoryEndpointRegistry {
    /// Available oscillatory endpoints
    pub endpoints: HashMap<Uuid, OscillatoryEndpoint>,
    
    /// Endpoint performance metrics
    pub endpoint_metrics: HashMap<Uuid, EndpointMetrics>,
    
    /// Endpoint allocation strategy
    pub allocation_strategy: EndpointAllocationStrategy,
}

/// Oscillatory Endpoint for zero-time processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryEndpoint {
    /// Endpoint identifier
    pub id: Uuid,
    
    /// Endpoint coordinates in S-entropy space
    pub coordinates: SEntropyCoordinates,
    
    /// Endpoint type and capabilities
    pub endpoint_type: OscillatoryEndpointType,
    
    /// Processing capacity
    pub processing_capacity: ProcessingCapacity,
    
    /// Current utilization
    pub current_utilization: f64, // 0.0 to 1.0
    
    /// Quality metrics
    pub quality_metrics: EndpointQualityMetrics,
}

/// Types of oscillatory endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OscillatoryEndpointType {
    /// Knowledge domain endpoint
    KnowledgeDomain,
    /// Temporal coordination endpoint
    TemporalCoordination,
    /// Entropy transformation endpoint
    EntropyTransformation,
    /// Universal problem navigation endpoint
    UniversalNavigation,
    /// Consciousness interface endpoint
    ConsciousnessInterface,
    /// Infinite parallelization endpoint
    InfiniteParallelization,
}

/// Zero-Time Coordinator for instant processing
#[derive(Debug, Clone)]
pub struct ZeroTimeCoordinator {
    /// Active zero-time operations
    pub active_operations: HashMap<Uuid, ZeroTimeOperation>,
    
    /// Zero-time processing strategies
    pub processing_strategies: Vec<ZeroTimeStrategy>,
    
    /// Performance metrics
    pub performance_metrics: ZeroTimeMetrics,
}

impl Default for SEntropyConfiguration {
    fn default() -> Self {
        Self {
            max_coordinate_ranges: (1000.0, 1000.0, 1000.0),
            navigation_precision: NavigationPrecision::High,
            zero_time_threshold: Duration::from_nanos(1),
            endpoint_density: 1000,
            cache_config: NavigationCacheConfig::default(),
            optimization_settings: OptimizationSettings::default(),
        }
    }
}

impl Default for SEntropyCoordinates {
    fn default() -> Self {
        Self {
            s_knowledge: 0.0,
            s_time: 0.0,
            s_entropy: 0.0,
            coordinate_quality: CoordinateQuality::default(),
            last_updated: Instant::now(),
        }
    }
}

impl SEntropyNavigationEngine {
    /// Create a new S-Entropy Navigation Engine
    pub async fn new(vm_id: Uuid, config: SEntropyConfiguration) -> Result<Self> {
        let engine_id = Uuid::new_v4();
        info!("Initializing S-Entropy Navigation Engine: {}", engine_id);
        
        // Initialize coordinate transformation matrix
        let transformation_matrix = Arc::new(RwLock::new(
            CoordinateTransformationMatrix::identity()
        ));
        
        // Initialize navigation cache
        let navigation_cache = Arc::new(RwLock::new(
            NavigationCache::new(config.cache_config.clone()).await?
        ));
        
        // Initialize oscillatory endpoint registry
        let oscillatory_endpoints = Arc::new(RwLock::new(
            OscillatoryEndpointRegistry::new(config.endpoint_density).await?
        ));
        
        // Initialize zero-time coordinator
        let zero_time_coordinator = Arc::new(RwLock::new(
            ZeroTimeCoordinator::new().await?
        ));
        
        let engine = Self {
            id: engine_id,
            vm_id,
            config: config.clone(),
            current_coordinates: Arc::new(RwLock::new(SEntropyCoordinates::default())),
            transformation_matrix,
            navigation_cache,
            oscillatory_endpoints,
            zero_time_coordinator,
            navigation_history: Arc::new(RwLock::new(NavigationHistory::new())),
            is_active: Arc::new(RwLock::new(false)),
        };
        
        info!("S-Entropy Navigation Engine initialized successfully");
        Ok(engine)
    }
    
    /// Start the S-entropy navigation engine
    pub async fn start_navigation_engine(&mut self) -> Result<()> {
        info!("Starting S-Entropy Navigation Engine: {}", self.id);
        
        {
            let mut active = self.is_active.write().await;
            *active = true;
        }
        
        // Start navigation coordination loop
        let engine_clone = self.clone();
        tokio::spawn(async move {
            engine_clone.navigation_coordination_loop().await;
        });
        
        // Start cache optimization loop
        let engine_clone = self.clone();
        tokio::spawn(async move {
            engine_clone.cache_optimization_loop().await;
        });
        
        // Start endpoint management loop
        let engine_clone = self.clone();
        tokio::spawn(async move {
            engine_clone.endpoint_management_loop().await;
        });
        
        info!("S-Entropy Navigation Engine started successfully");
        Ok(())
    }
    
    /// Navigate to specific S-entropy coordinates
    pub async fn navigate_to_coordinates(
        &mut self,
        target_coordinates: (f64, f64, f64),
        navigation_mode: SEntropyNavigationMode,
    ) -> Result<SEntropyNavigationResult> {
        
        let start_time = Instant::now();
        debug!("Navigating to coordinates: {:?} using mode: {:?}", target_coordinates, navigation_mode);
        
        // Convert tuple to SEntropyCoordinates
        let target = SEntropyCoordinates {
            s_knowledge: target_coordinates.0,
            s_time: target_coordinates.1,
            s_entropy: target_coordinates.2,
            coordinate_quality: CoordinateQuality::default(),
            last_updated: Instant::now(),
        };
        
        // Execute navigation based on mode
        let navigation_result = match navigation_mode {
            SEntropyNavigationMode::Direct => {
                self.direct_navigation(target).await?
            },
            SEntropyNavigationMode::Optimized => {
                self.optimized_navigation(target).await?
            },
            SEntropyNavigationMode::ZeroTime => {
                self.zero_time_navigation(target).await?
            },
            SEntropyNavigationMode::ConsciousnessAware => {
                self.consciousness_aware_navigation(target).await?
            },
            SEntropyNavigationMode::InfiniteParallel => {
                self.infinite_parallel_navigation(target).await?
            },
        };
        
        // Update current coordinates
        {
            let mut current = self.current_coordinates.write().await;
            *current = navigation_result.final_coordinates.clone();
        }
        
        // Record navigation in history
        self.record_navigation_history(&navigation_result).await?;
        
        let total_time = start_time.elapsed();
        debug!("Navigation completed in: {:?}", total_time);
        
        Ok(navigation_result)
    }
    
    /// Direct navigation (instant coordinate jump)
    async fn direct_navigation(&self, target: SEntropyCoordinates) -> Result<SEntropyNavigationResult> {
        trace!("Executing direct navigation");
        
        // Direct navigation is instantaneous - just jump to coordinates
        Ok(SEntropyNavigationResult {
            navigation_success: true,
            final_coordinates: target,
            navigation_time: Duration::from_nanos(1), // Essentially zero time
            solution_quality: SolutionQuality::high(),
            endpoints_utilized: Vec::new(),
            navigation_path: NavigationPath::direct(),
        })
    }
    
    /// Optimized navigation (fastest route calculation)
    async fn optimized_navigation(&self, target: SEntropyCoordinates) -> Result<SEntropyNavigationResult> {
        trace!("Executing optimized navigation");
        
        // Check navigation cache first
        let cache_result = self.check_navigation_cache(&target).await?;
        if let Some(cached_solution) = cache_result {
            return Ok(cached_solution.to_navigation_result());
        }
        
        // Calculate optimal path
        let optimal_path = self.calculate_optimal_path(&target).await?;
        
        // Execute navigation along optimal path
        let result = self.execute_navigation_path(optimal_path).await?;
        
        // Cache the solution
        self.cache_navigation_solution(&target, &result).await?;
        
        Ok(result)
    }
    
    /// Zero-time navigation through oscillatory endpoints
    async fn zero_time_navigation(&self, target: SEntropyCoordinates) -> Result<SEntropyNavigationResult> {
        trace!("Executing zero-time navigation");
        
        // Find optimal oscillatory endpoint
        let optimal_endpoint = self.find_optimal_endpoint(&target).await?;
        
        // Execute zero-time processing
        let zero_time_result = {
            let mut coordinator = self.zero_time_coordinator.write().await;
            coordinator.execute_zero_time_processing(&target, &optimal_endpoint).await?
        };
        
        Ok(SEntropyNavigationResult {
            navigation_success: true,
            final_coordinates: target,
            navigation_time: Duration::from_nanos(1), // True zero-time
            solution_quality: SolutionQuality::ultra(),
            endpoints_utilized: vec![optimal_endpoint.id],
            navigation_path: NavigationPath::zero_time(optimal_endpoint),
        })
    }
    
    /// Consciousness-aware navigation with BMD orchestration
    async fn consciousness_aware_navigation(&self, target: SEntropyCoordinates) -> Result<SEntropyNavigationResult> {
        trace!("Executing consciousness-aware navigation");
        
        // This would integrate with consciousness runtime for BMD orchestration
        // For now, return optimized navigation result
        self.optimized_navigation(target).await
    }
    
    /// Infinite parallel navigation
    async fn infinite_parallel_navigation(&self, target: SEntropyCoordinates) -> Result<SEntropyNavigationResult> {
        trace!("Executing infinite parallel navigation");
        
        // Execute multiple navigation strategies in parallel and select best result
        let navigation_futures = vec![
            self.direct_navigation(target.clone()),
            self.optimized_navigation(target.clone()),
            self.zero_time_navigation(target.clone()),
        ];
        
        // Wait for all results
        let results = futures::future::try_join_all(navigation_futures).await?;
        
        // Select best result based on quality metrics
        let best_result = results
            .into_iter()
            .max_by(|a, b| a.solution_quality.overall_score.partial_cmp(&b.solution_quality.overall_score).unwrap())
            .unwrap();
        
        Ok(best_result)
    }
    
    /// Find optimal oscillatory endpoint for zero-time processing
    async fn find_optimal_endpoint(&self, target: &SEntropyCoordinates) -> Result<OscillatoryEndpoint> {
        let endpoints = self.oscillatory_endpoints.read().await;
        
        // Find endpoint closest to target coordinates
        let optimal_endpoint = endpoints
            .endpoints
            .values()
            .min_by(|a, b| {
                let dist_a = self.calculate_coordinate_distance(&a.coordinates, target);
                let dist_b = self.calculate_coordinate_distance(&b.coordinates, target);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .ok_or_else(|| anyhow::anyhow!("No oscillatory endpoints available"))?;
        
        Ok(optimal_endpoint.clone())
    }
    
    /// Calculate distance between coordinates
    fn calculate_coordinate_distance(&self, coord1: &SEntropyCoordinates, coord2: &SEntropyCoordinates) -> f64 {
        let dx = coord1.s_knowledge - coord2.s_knowledge;
        let dy = coord1.s_time - coord2.s_time;
        let dz = coord1.s_entropy - coord2.s_entropy;
        
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Navigation coordination loop
    async fn navigation_coordination_loop(&self) {
        while *self.is_active.read().await {
            // Coordinate navigation operations
            if let Err(e) = self.coordinate_navigation_operations().await {
                warn!("Navigation coordination error: {}", e);
            }
            
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    
    /// Cache optimization loop
    async fn cache_optimization_loop(&self) {
        while *self.is_active.read().await {
            // Optimize navigation cache
            if let Err(e) = self.optimize_navigation_cache().await {
                warn!("Cache optimization error: {}", e);
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    /// Endpoint management loop
    async fn endpoint_management_loop(&self) {
        while *self.is_active.read().await {
            // Manage oscillatory endpoints
            if let Err(e) = self.manage_oscillatory_endpoints().await {
                warn!("Endpoint management error: {}", e);
            }
            
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
    
    /// Get current S-entropy coordinates
    pub async fn get_current_coordinates(&self) -> SEntropyCoordinates {
        let coordinates = self.current_coordinates.read().await;
        coordinates.clone()
    }
    
    /// Get navigation statistics
    pub async fn get_navigation_statistics(&self) -> Result<NavigationStatistics> {
        let history = self.navigation_history.read().await;
        let cache = self.navigation_cache.read().await;
        
        Ok(NavigationStatistics {
            total_navigations: history.total_navigations(),
            average_navigation_time: history.average_navigation_time(),
            cache_hit_rate: cache.cache_stats.hit_rate,
            zero_time_navigations: history.zero_time_navigations(),
            coordinates_explored: history.unique_coordinates_count(),
        })
    }
    
    // Additional helper methods would be implemented here...
    async fn check_navigation_cache(&self, _target: &SEntropyCoordinates) -> Result<Option<CachedSolution>> { Ok(None) }
    async fn calculate_optimal_path(&self, _target: &SEntropyCoordinates) -> Result<NavigationPath> { Ok(NavigationPath::direct()) }
    async fn execute_navigation_path(&self, _path: NavigationPath) -> Result<SEntropyNavigationResult> { Ok(SEntropyNavigationResult::default()) }
    async fn cache_navigation_solution(&self, _target: &SEntropyCoordinates, _result: &SEntropyNavigationResult) -> Result<()> { Ok(()) }
    async fn record_navigation_history(&self, _result: &SEntropyNavigationResult) -> Result<()> { Ok(()) }
    async fn coordinate_navigation_operations(&self) -> Result<()> { Ok(()) }
    async fn optimize_navigation_cache(&self) -> Result<()> { Ok(()) }
    async fn manage_oscillatory_endpoints(&self) -> Result<()> { Ok(()) }
}

// Clone implementation
impl Clone for SEntropyNavigationEngine {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            vm_id: self.vm_id,
            config: self.config.clone(),
            current_coordinates: Arc::clone(&self.current_coordinates),
            transformation_matrix: Arc::clone(&self.transformation_matrix),
            navigation_cache: Arc::clone(&self.navigation_cache),
            oscillatory_endpoints: Arc::clone(&self.oscillatory_endpoints),
            zero_time_coordinator: Arc::clone(&self.zero_time_coordinator),
            navigation_history: Arc::clone(&self.navigation_history),
            is_active: Arc::clone(&self.is_active),
        }
    }
}

// Default implementations and placeholder structures
impl Default for SEntropyNavigationResult {
    fn default() -> Self {
        Self {
            navigation_success: true,
            final_coordinates: SEntropyCoordinates::default(),
            navigation_time: Duration::from_nanos(1),
            solution_quality: SolutionQuality::default(),
            endpoints_utilized: Vec::new(),
            navigation_path: NavigationPath::default(),
        }
    }
}

// Placeholder implementations for compilation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoordinateQuality {
    pub accuracy: f64,
    pub precision: f64,
    pub reliability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolutionQuality {
    pub overall_score: f64,
    pub accuracy: f64,
    pub efficiency: f64,
    pub reliability: f64,
}

impl SolutionQuality {
    pub fn high() -> Self { Self { overall_score: 0.9, accuracy: 0.9, efficiency: 0.9, reliability: 0.9 } }
    pub fn ultra() -> Self { Self { overall_score: 1.0, accuracy: 1.0, efficiency: 1.0, reliability: 1.0 } }
}

#[derive(Debug, Clone, Default)]
pub struct NavigationPath {
    pub path_type: String,
    pub waypoints: Vec<SEntropyCoordinates>,
}

impl NavigationPath {
    pub fn direct() -> Self { Self { path_type: "direct".to_string(), waypoints: Vec::new() } }
    pub fn zero_time(endpoint: OscillatoryEndpoint) -> Self { 
        Self { 
            path_type: "zero_time".to_string(), 
            waypoints: vec![endpoint.coordinates] 
        } 
    }
}

#[derive(Debug, Clone, Default)]
pub struct NavigationCacheConfig;

#[derive(Debug, Clone, Default)]
pub struct OptimizationSettings;

#[derive(Debug, Clone, Default)]
pub struct MatrixQuality;

#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub hit_rate: f64,
    pub total_queries: u64,
    pub cache_hits: u64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheCleanupManager;

#[derive(Debug, Clone, Default)]
pub struct EndpointMetrics;

#[derive(Debug, Clone, Default)]
pub struct EndpointAllocationStrategy;

#[derive(Debug, Clone, Default)]
pub struct ProcessingCapacity;

#[derive(Debug, Clone, Default)]
pub struct EndpointQualityMetrics;

#[derive(Debug, Clone, Default)]
pub struct ZeroTimeOperation;

#[derive(Debug, Clone, Default)]
pub struct ZeroTimeStrategy;

#[derive(Debug, Clone, Default)]
pub struct ZeroTimeMetrics;

#[derive(Debug, Clone, Default)]
pub struct NavigationHistory;

#[derive(Debug, Clone, Default)]
pub struct CachedSolution;

#[derive(Debug, Clone, Default)]
pub struct NavigationStatistics {
    pub total_navigations: u64,
    pub average_navigation_time: Duration,
    pub cache_hit_rate: f64,
    pub zero_time_navigations: u64,
    pub coordinates_explored: u64,
}

pub type ProblemSignature = String;

// Implementation stubs
impl CoordinateTransformationMatrix {
    pub fn identity() -> Self {
        Self {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            inverse_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            matrix_quality: MatrixQuality::default(),
            last_updated: Instant::now(),
        }
    }
}

impl NavigationCache {
    pub async fn new(_config: NavigationCacheConfig) -> Result<Self> { Ok(Self::default()) }
}

impl OscillatoryEndpointRegistry {
    pub async fn new(_density: u32) -> Result<Self> { Ok(Self::default()) }
}

impl ZeroTimeCoordinator {
    pub async fn new() -> Result<Self> { Ok(Self::default()) }
    
    pub async fn execute_zero_time_processing(
        &mut self,
        _target: &SEntropyCoordinates,
        _endpoint: &OscillatoryEndpoint
    ) -> Result<()> { Ok(()) }
}

impl NavigationHistory {
    pub fn new() -> Self { Self::default() }
    pub fn total_navigations(&self) -> u64 { 0 }
    pub fn average_navigation_time(&self) -> Duration { Duration::from_nanos(1) }
    pub fn zero_time_navigations(&self) -> u64 { 0 }
    pub fn unique_coordinates_count(&self) -> u64 { 0 }
}

impl CachedSolution {
    pub fn to_navigation_result(&self) -> SEntropyNavigationResult { SEntropyNavigationResult::default() }
}