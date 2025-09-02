'use client'

import { useState } from 'react'

interface TabNavigationProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

export default function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  const tabs = [
    { id: 'upload', label: 'Dataset Upload', icon: 'UPLOAD' },
    { id: 'training', label: 'ML Training', icon: 'TRAIN' },
    { id: 'inference', label: 'Testing', icon: 'TEST' }
  ]

  return (
    <div className="flex justify-center mb-8">
      <div className="flex bg-gray-900/50 rounded-lg p-1 backdrop-blur-sm">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex items-center px-4 py-2 rounded-md transition-all duration-300 ${
              activeTab === tab.id
                ? 'bg-white text-black shadow-lg transform scale-105'
                : 'text-gray-300 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <span className="mr-2 text-xs font-bold bg-gray-700 px-2 py-1 rounded">{tab.icon}</span>
            <span className="font-medium">{tab.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
