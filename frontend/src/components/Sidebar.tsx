import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard, Users, FlaskConical, BarChart3,
  BrainCircuit, Settings, BookOpen, Activity
} from 'lucide-react'

const nav = [
  { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/patients',  icon: Users,           label: 'Analisi Paziente' },
  { to: '/batch',     icon: FlaskConical,    label: 'Analisi Batch' },
  { to: '/results',   icon: BarChart3,       label: 'Risultati' },
  { to: '/models',    icon: BrainCircuit,    label: 'Gestione Modelli' },
  { to: '/settings',  icon: Settings,        label: 'Impostazioni' },
]

export default function Sidebar() {
  return (
    <aside className="w-64 flex-shrink-0 bg-gradient-to-b from-navy-700 to-navy-500 flex flex-col">
      {/* Logo */}
      <div className="px-5 py-6 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-white/15 flex items-center justify-center">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-white font-bold text-sm leading-tight">LLM Dementia</p>
            <p className="text-white/50 text-xs">Diagnosis Support</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        <p className="px-3 text-white/40 text-xs font-semibold uppercase tracking-wider mb-2">
          Navigazione
        </p>
        {nav.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `sidebar-item ${isActive ? 'active' : ''}`
            }
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/10">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-white/40" />
          <span className="text-white/40 text-xs">v1.0.0 — Ricerca</span>
        </div>
      </div>
    </aside>
  )
}
