/**
 * Utility functions and types for CyberShield frontend
 */

/**
 * Format risk score as percentage
 */
export const formatRiskScore = (score: number): string => {
    return `${(score * 100).toFixed(1)}%`;
};

/**
 * Get risk level badge color based on score
 */
export const getRiskBadgeColor = (score: number): string => {
    if (score < 0.25) return 'bg-emerald-100 text-emerald-800';
    if (score < 0.5) return 'bg-amber-100 text-amber-800';
    if (score < 0.75) return 'bg-orange-100 text-orange-800';
    return 'bg-red-100 text-red-800';
};

/**
 * Get risk level label
 */
export const getRiskLevel = (score: number): string => {
    if (score < 0.25) return 'Safe';
    if (score < 0.5) return 'Low Risk';
    if (score < 0.75) return 'Medium Risk';
    return 'High Risk';
};

/**
 * Get risk indicator color for charts/visualizations
 */
export const getRiskColor = (score: number): string => {
    if (score < 0.25) return '#10b981'; // Emerald
    if (score < 0.5) return '#f59e0b'; // Amber
    if (score < 0.75) return '#f97316'; // Orange
    return '#ef4444'; // Red
};

/**
 * Format bytes for display
 */
export const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';

    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Format timestamp to readable date
 */
export const formatDate = (timestamp: number | string): string => {
    const date = new Date(typeof timestamp === 'string' ? timestamp : timestamp * 1000);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};

/**
 * Format duration in seconds to readable format
 */
export const formatDuration = (seconds: number): string => {
    if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
};

/**
 * Format percentage with proper decimal places
 */
export const formatPercent = (value: number): string => {
    return `${(value * 100).toFixed(2)}%`;
};

/**
 * Convert file to base64 for API submission
 */
export const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const result = reader.result as string;
            resolve(result.split(',')[1]);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
};

/**
 * Validate file type
 */
export const isValidFileType = (file: File, allowedTypes: string[]): boolean => {
    return allowedTypes.includes(file.type) ||
        allowedTypes.some(type => file.name.endsWith(type));
};

/**
 * Sleep/delay utility for async operations
 */
export const sleep = (ms: number): Promise<void> => {
    return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Debounce utility
 */
export const debounce = <T extends any[], R>(
    fn: (...args: T) => R,
    delay: number
): ((...args: T) => void) => {
    let timeoutId: NodeJS.Timeout;
    return (...args: T) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn(...args), delay);
    };
};

/**
 * Throttle utility
 */
export const throttle = <T extends any[], R>(
    fn: (...args: T) => R,
    limit: number
): ((...args: T) => void) => {
    let inThrottle: boolean;
    return (...args: T) => {
        if (!inThrottle) {
            fn(...args);
            inThrottle = true;
            setTimeout(() => (inThrottle = false), limit);
        }
    };
};

/**
 * Clamp value between min and max
 */
export const clamp = (value: number, min: number, max: number): number => {
    return Math.max(min, Math.min(max, value));
};

/**
 * Generate color based on value in range [0, 1]
 */
export const getGradientColor = (value: number): string => {
    const colors = [
        '#10b981', // green - 0
        '#84cc16', // lime
        '#eab308', // yellow
        '#f97316', // orange
        '#ef4444', // red - 1
    ];

    const clampedValue = clamp(value, 0, 1);
    const index = clampedValue * (colors.length - 1);
    const floorIndex = Math.floor(index);
    const ceilIndex = Math.ceil(index);
    const fraction = index - floorIndex;

    if (floorIndex === ceilIndex) {
        return colors[floorIndex];
    }

    // Linear interpolation would be complex for hex colors, just return closest
    return fraction < 0.5 ? colors[floorIndex] : colors[ceilIndex];
};

/**
 * Type-safe environment variable getter
 */
export const getEnv = (key: string, defaultValue?: string): string => {
    return process.env[`NEXT_PUBLIC_${key}`] || defaultValue || '';
};
