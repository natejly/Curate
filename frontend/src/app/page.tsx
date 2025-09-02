'use client'

import { useState } from 'react'
import CursorAnimation from '@/components/CursorAnimation'
import AnimatedBackground from '@/components/AnimatedBackground'
import TabNavigation from '@/components/TabNavigation'
import DataUploadTab from '@/components/DataUploadTab'
import MLTrainingTab from '@/components/MLTrainingTab'
import TestingTab from '@/components/TestingTab'

export default function Home() {
  const [activeTab, setActiveTab] = useState('upload')

  const renderTabContent = () => {
    switch (activeTab) {
      case 'upload':
        return <DataUploadTab />
      case 'training':
        return <MLTrainingTab />
      case 'inference':
        return <TestingTab />
      default:
        return <DataUploadTab />
    }
  }

  return (
    <main className="min-h-screen bg-black overflow-hidden">
      <AnimatedBackground />
      <CursorAnimation />

      <div className="relative z-20 container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">
            Curate AI Platform
          </h1>
          <p className="text-gray-300 text-lg max-w-2xl mx-auto">
            Upload data, train models, and test inferences with our comprehensive AI platform
          </p>
        </div>

        {/* Tab Navigation */}
        <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab Content */}
        <div className="bg-black/30 backdrop-blur-sm rounded-xl border border-gray-800 p-8">
          {renderTabContent()}
        </div>
      </div>
    </main>
  )
}
