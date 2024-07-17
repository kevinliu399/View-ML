import React, { useContext, createContext, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronLast, ChevronFirst } from 'lucide-react';

const SidebarContext = createContext();

export default function Sidebar({ children }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <aside className="h-screen">
      <nav className="h-full flex flex-col bg-white border-r shadow-sm">
        <div className="p-4 pb-2 flex justify-between items-center">
          <h1 className={`overflow-hidden transition-all font-bold text-xl  ${expanded ? "w-46 py-4" : "w-0"}`}>My Models Collection</h1>
          <button
            onClick={() => setExpanded((curr) => !curr)}
            className="p-1.5 rounded-lg bg-gray-50 hover:bg-gray-100 outline outline-1 shadow-md outline-red-200"
          >
            {expanded ? <ChevronFirst /> : <ChevronLast />}
          </button>
        </div>
        <SidebarContext.Provider value={{ expanded }}>
          <ul className="flex-1 px-3">{children}</ul>
        </SidebarContext.Provider>
      </nav>
    </aside>
  );
}

export function SidebarItem({ icon, text, to }) {
  const { expanded } = useContext(SidebarContext);
  const location = useLocation();
  const isActive = location.pathname === to;

  return (
    <li
      className={`
        relative flex items-center py-2 px-3 my-1
        font-medium rounded-md cursor-pointer
        transition-colors group
        ${isActive ? "bg-gradient-to-tr from-red-200 to-red-100 text-red-800" : "hover:bg-red-50 text-gray-600"}
      `}
    >
      {icon}
      <Link to={to} className={`overflow-hidden transition-all ${expanded ? "w-52 ml-3" : "w-0"}`}>
        {text}
      </Link>
    </li>
  );
}
