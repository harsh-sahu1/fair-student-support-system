# Student Support AI — Frontend

Educator-facing decision-support user interface built with React, TypeScript, and Vite.

## Design Philosophy
Designed specifically for school counselors and academic administrators allocating limited support capacity.
- **Calm, Distraction-Free Palette**: Neutral slate canvas (#F8FAFC), deep slate typography (#0F172A), and single warm amber accent (#D97706) reserved strictly for priority intervention status.
- **No Generic AI Dashboard Tells**: Zero gradients, zero neon, zero unreadable monospace numbers, no ambiguous cards.
- **Visual Demographic Parity**: Group recall rates presented with explicit Observed Recall Gap annotations and eligibility indicators.
- **Ethical Safeguards**: Clear non-causal language and mandatory human-in-the-loop review notices.

## Setup & Running

`ash
npm install
npm run dev
`

The frontend will run on [http://localhost:5173](http://localhost:5173).

Configure the backend URL in .env:
`
VITE_API_BASE_URL=http://127.0.0.1:8000
`
