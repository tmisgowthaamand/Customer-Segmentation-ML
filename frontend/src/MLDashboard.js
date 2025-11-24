import React, { useState } from 'react';
import axios from 'axios';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Label } from './components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { toast } from 'sonner';

const API_URL = process.env.REACT_APP_BACKEND_URL;

function MLDashboard() {
  const [customerData, setCustomerData] = useState({
    age: 35,
    gender: 'Male',
    income: 65000,
    spending_score: 50,
    region: 'North',
    purchase_frequency: 12,
    avg_order_value: 500,
    recency: 30
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (field, value) => {
    setCustomerData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/api/predict_cluster`, customerData);
      setPrediction(response.data);
      toast.success('Prediction completed successfully!');
    } catch (error) {
      console.error('Prediction error:', error);
      toast.error('Failed to predict cluster');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                Customer Segmentation ML
              </h1>
              <p className="text-gray-600 mt-1">AI-Powered Customer Analytics with K-Means Clustering</p>
            </div>
            <div className="flex gap-3">
              <a 
                href="http://localhost:8501" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                <Button data-testid="open-streamlit-btn" className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700">
                  <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Open Full Dashboard
                </Button>
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <Card data-testid="customer-input-card" className="shadow-xl border-0 bg-white/80 backdrop-blur">
            <CardHeader className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-t-lg">
              <CardTitle className="text-2xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                Customer Details
              </CardTitle>
              <CardDescription className="text-purple-100">
                Enter customer information to predict their segment
              </CardDescription>
            </CardHeader>
            <CardContent className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="age" className="text-sm font-semibold text-gray-700">Age</Label>
                  <Input
                    id="age"
                    data-testid="age-input"
                    type="number"
                    value={customerData.age}
                    onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                    min="18"
                    max="100"
                    className="mt-1"
                  />
                </div>
                
                <div>
                  <Label htmlFor="gender" className="text-sm font-semibold text-gray-700">Gender</Label>
                  <Select value={customerData.gender} onValueChange={(value) => handleInputChange('gender', value)}>
                    <SelectTrigger data-testid="gender-select" className="mt-1">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Male">Male</SelectItem>
                      <SelectItem value="Female">Female</SelectItem>
                      <SelectItem value="Other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="income" className="text-sm font-semibold text-gray-700">Annual Income ($)</Label>
                <Input
                  id="income"
                  data-testid="income-input"
                  type="number"
                  value={customerData.income}
                  onChange={(e) => handleInputChange('income', parseInt(e.target.value))}
                  min="20000"
                  step="1000"
                  className="mt-1"
                />
              </div>

              <div>
                <Label htmlFor="spending" className="text-sm font-semibold text-gray-700">
                  Spending Score (1-100): {customerData.spending_score}
                </Label>
                <input
                  id="spending"
                  data-testid="spending-input"
                  type="range"
                  min="1"
                  max="100"
                  value={customerData.spending_score}
                  onChange={(e) => handleInputChange('spending_score', parseInt(e.target.value))}
                  className="w-full mt-2 accent-purple-600"
                />
              </div>

              <div>
                <Label htmlFor="region" className="text-sm font-semibold text-gray-700">Region</Label>
                <Select value={customerData.region} onValueChange={(value) => handleInputChange('region', value)}>
                  <SelectTrigger data-testid="region-select" className="mt-1">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="North">North</SelectItem>
                    <SelectItem value="South">South</SelectItem>
                    <SelectItem value="East">East</SelectItem>
                    <SelectItem value="West">West</SelectItem>
                    <SelectItem value="Central">Central</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label htmlFor="frequency" className="text-sm font-semibold text-gray-700">Purchase Frequency/Year</Label>
                  <Input
                    id="frequency"
                    data-testid="frequency-input"
                    type="number"
                    value={customerData.purchase_frequency}
                    onChange={(e) => handleInputChange('purchase_frequency', parseInt(e.target.value))}
                    min="0"
                    className="mt-1"
                  />
                </div>
                
                <div>
                  <Label htmlFor="aov" className="text-sm font-semibold text-gray-700">Avg Order Value ($)</Label>
                  <Input
                    id="aov"
                    data-testid="aov-input"
                    type="number"
                    value={customerData.avg_order_value}
                    onChange={(e) => handleInputChange('avg_order_value', parseInt(e.target.value))}
                    min="0"
                    className="mt-1"
                  />
                </div>
              </div>

              <div>
                <Label htmlFor="recency" className="text-sm font-semibold text-gray-700">Recency (Days Since Last Purchase)</Label>
                <Input
                  id="recency"
                  data-testid="recency-input"
                  type="number"
                  value={customerData.recency}
                  onChange={(e) => handleInputChange('recency', parseInt(e.target.value))}
                  min="0"
                  max="365"
                  className="mt-1"
                />
              </div>

              <Button
                data-testid="predict-btn"
                onClick={handlePredict}
                disabled={loading}
                className="w-full mt-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white font-semibold py-6 text-lg"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </div>
                ) : (
                  <span className="flex items-center justify-center">
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Predict Customer Segment
                  </span>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          <div className="space-y-6">
            {prediction ? (
              <>
                <Card data-testid="prediction-result-card" className="shadow-xl border-0 bg-gradient-to-br from-purple-600 to-indigo-600 text-white">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-3xl font-bold" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                        Cluster {prediction.cluster}
                      </CardTitle>
                      <div className="bg-white/20 backdrop-blur rounded-full px-4 py-2">
                        <span className="text-sm font-semibold">Segment ID</span>
                      </div>
                    </div>
                    <CardDescription className="text-purple-100 text-lg mt-2">
                      Customer belongs to this segment based on their profile
                    </CardDescription>
                  </CardHeader>
                </Card>

                <Card data-testid="cluster-stats-card" className="shadow-xl border-0 bg-white/80 backdrop-blur">
                  <CardHeader>
                    <CardTitle className="text-xl" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                      Cluster Characteristics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
                        <p className="text-sm text-blue-600 font-semibold">Cluster Size</p>
                        <p className="text-2xl font-bold text-blue-900 mt-1">{prediction.cluster_size}</p>
                        <p className="text-xs text-blue-600 mt-1">customers</p>
                      </div>

                      <div className="bg-gradient-to-br from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
                        <p className="text-sm text-green-600 font-semibold">Avg Income</p>
                        <p className="text-2xl font-bold text-green-900 mt-1">
                          ${Math.round(prediction.cluster_characteristics.avg_income).toLocaleString()}
                        </p>
                        <p className="text-xs text-green-600 mt-1">annual</p>
                      </div>

                      <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
                        <p className="text-sm text-purple-600 font-semibold">Avg Spending Score</p>
                        <p className="text-2xl font-bold text-purple-900 mt-1">
                          {prediction.cluster_characteristics.avg_spending_score.toFixed(1)}
                        </p>
                        <p className="text-xs text-purple-600 mt-1">out of 100</p>
                      </div>

                      <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-4 rounded-lg border border-orange-200">
                        <p className="text-sm text-orange-600 font-semibold">Avg Total Spend</p>
                        <p className="text-2xl font-bold text-orange-900 mt-1">
                          ${Math.round(prediction.cluster_characteristics.avg_total_spend).toLocaleString()}
                        </p>
                        <p className="text-xs text-orange-600 mt-1">annually</p>
                      </div>

                      <div className="bg-gradient-to-br from-pink-50 to-pink-100 p-4 rounded-lg border border-pink-200">
                        <p className="text-sm text-pink-600 font-semibold">Avg Purchase Frequency</p>
                        <p className="text-2xl font-bold text-pink-900 mt-1">
                          {prediction.cluster_characteristics.avg_purchase_frequency.toFixed(1)}
                        </p>
                        <p className="text-xs text-pink-600 mt-1">per year</p>
                      </div>

                      <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 rounded-lg border border-indigo-200">
                        <p className="text-sm text-indigo-600 font-semibold">Avg Recency</p>
                        <p className="text-2xl font-bold text-indigo-900 mt-1">
                          {Math.round(prediction.cluster_characteristics.avg_recency)}
                        </p>
                        <p className="text-xs text-indigo-600 mt-1">days ago</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card data-testid="marketing-recommendation-card" className="shadow-xl border-0 bg-white/80 backdrop-blur">
                  <CardHeader className="bg-gradient-to-r from-amber-500 to-orange-500 text-white rounded-t-lg">
                    <CardTitle style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
                      Marketing Recommendations
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-6">
                    {prediction.cluster === 0 ? (
                      <div className="space-y-3">
                        <div className="flex items-start">
                          <div className="bg-blue-100 rounded-full p-2 mr-3">
                            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">Budget-Conscious Segment</h4>
                            <p className="text-sm text-gray-600 mt-1">Focus on discount campaigns, promotions, and value-based messaging</p>
                          </div>
                        </div>
                        <div className="flex items-start">
                          <div className="bg-green-100 rounded-full p-2 mr-3">
                            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">Recommended Strategy</h4>
                            <p className="text-sm text-gray-600 mt-1">Loyalty programs, bundle offers, and seasonal discounts to increase engagement</p>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="flex items-start">
                          <div className="bg-purple-100 rounded-full p-2 mr-3">
                            <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">Premium Customer Segment</h4>
                            <p className="text-sm text-gray-600 mt-1">Target with exclusive offers, VIP programs, and premium product launches</p>
                          </div>
                        </div>
                        <div className="flex items-start">
                          <div className="bg-amber-100 rounded-full p-2 mr-3">
                            <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                          </div>
                          <div>
                            <h4 className="font-semibold text-gray-900">Recommended Strategy</h4>
                            <p className="text-sm text-gray-600 mt-1">Personalized experiences, early access to new products, and loyalty rewards</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </>
            ) : (
              <Card className="shadow-xl border-0 bg-white/80 backdrop-blur h-full">
                <CardContent className="flex flex-col items-center justify-center h-full p-12 text-center">
                  <div className="bg-gradient-to-br from-purple-100 to-indigo-100 rounded-full p-6 mb-4">
                    <svg className="w-16 h-16 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">No Prediction Yet</h3>
                  <p className="text-gray-600 max-w-md">
                    Enter customer details on the left and click "Predict Customer Segment" to see AI-powered segmentation results
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <Card className="border-0 bg-white/80 backdrop-blur shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-start">
                <div className="bg-blue-100 rounded-lg p-3 mr-4">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">K-Means Clustering</h3>
                  <p className="text-sm text-gray-600">Advanced ML algorithm segments customers based on behavioral patterns</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 bg-white/80 backdrop-blur shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-start">
                <div className="bg-green-100 rounded-lg p-3 mr-4">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">Real-Time Predictions</h3>
                  <p className="text-sm text-gray-600">Instant customer segmentation powered by FastAPI backend</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-0 bg-white/80 backdrop-blur shadow-lg">
            <CardContent className="p-6">
              <div className="flex items-start">
                <div className="bg-purple-100 rounded-lg p-3 mr-4">
                  <svg className="w-6 h-6 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                  </svg>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 mb-1">Actionable Insights</h3>
                  <p className="text-sm text-gray-600">Get marketing recommendations tailored to each customer segment</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

export default MLDashboard;
