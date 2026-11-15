<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Perseus | SYBIL &amp; IRIS Planetary Analogue Discovery System</title>
    <style>
        /* Reset & Base */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.65;
            color: #e0f2fe;
            background-color: #030712;
            background-image: 
                radial-gradient(ellipse at top, #0c2442 0%, #050d1a 55%, #02040a 100%);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }

        /* Starfield Background */
        body::before,
        body::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        body::before {
            background-image:
                radial-gradient(1.5px 1.5px at 12% 18%, #ffffff, transparent),
                radial-gradient(1px 1px at 28% 42%, #bae6fd, transparent),
                radial-gradient(2px 2px at 45% 15%, #7dd3fc, transparent),
                radial-gradient(1px 1px at 63% 33%, #ffffff, transparent),
                radial-gradient(1.5px 1.5px at 78% 12%, #e0f2fe, transparent),
                radial-gradient(1px 1px at 89% 48%, #38bdf8, transparent),
                radial-gradient(1.5px 1.5px at 18% 72%, #ffffff, transparent),
                radial-gradient(2px 2px at 36% 84%, #bae6fd, transparent),
                radial-gradient(1px 1px at 54% 66%, #ffffff, transparent),
                radial-gradient(1.5px 1.5px at 71% 79%, #7dd3fc, transparent),
                radial-gradient(1px 1px at 92% 88%, #ffffff, transparent),
                radial-gradient(1.5px 1.5px at 8% 52%, #38bdf8, transparent);
            background-size: 550px 550px;
            opacity: 0.85;
        }

        body::after {
            background-image:
                radial-gradient(1px 1px at 22% 27%, #ffffff, transparent),
                radial-gradient(1.5px 1.5px at 49% 51%, #38bdf8, transparent),
                radial-gradient(1px 1px at 81% 63%, #bae6fd, transparent),
                radial-gradient(1px 1px at 67% 91%, #ffffff, transparent),
                radial-gradient(1.5px 1.5px at 15% 89%, #7dd3fc, transparent);
            background-size: 380px 380px;
            opacity: 0.55;
        }

        /* Header & Navigation */
        header {
            background-color: rgba(6, 16, 33, 0.9);
            border-bottom: 1px solid rgba(56, 189, 248, 0.25);
            padding: 1rem 5%;
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(12px);
        }

        nav {
            max-width: 1100px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            font-weight: 800;
            font-size: 1.25rem;
            color: #ffffff;
            text-decoration: none;
            letter-spacing: 0.8px;
            cursor: pointer;
        }

        .logo span {
            color: #38bdf8;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
        }

        .nav-links {
            list-style: none;
            display: flex;
            align-items: center;
            gap: 1.4rem;
        }

        .nav-links li a, .nav-links li button {
            text-decoration: none;
            background: none;
            border: none;
            color: #bae6fd;
            font-weight: 500;
            font-size: 0.95rem;
            cursor: pointer;
            font-family: inherit;
            transition: color 0.2s ease;
        }

        .nav-links li a:hover,
        .nav-links li a.active,
        .nav-links li button:hover,
        .nav-links li button.active {
            color: #38bdf8;
        }

        /* Dropdown Menu */
        .dropdown {
            position: relative;
        }

        .dropdown-toggle {
            background: rgba(14, 116, 144, 0.22) !important;
            color: #e0f2fe !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            padding: 0.48rem 1rem !important;
            border-radius: 8px !important;
            font-size: 0.94rem !important;
            font-weight: 600 !important;
            display: flex;
            align-items: center;
            gap: 0.45rem;
            transition: all 0.2s ease;
        }

        .dropdown-toggle:hover,
        .dropdown-toggle.active-parent {
            background: rgba(2, 132, 199, 0.35) !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
        }

        .arrow {
            font-size: 0.75rem;
            transition: transform 0.25s ease;
        }

        .dropdown.open .arrow {
            transform: rotate(180deg);
        }

        .dropdown-menu {
            position: absolute;
            right: 0;
            top: calc(100% + 10px);
            min-width: 285px;
            background-color: #071326;
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 10px;
            box-shadow: 0 18px 35px rgba(0, 0, 0, 0.65), 0 0 20px rgba(14, 165, 233, 0.15);
            list-style: none;
            padding: 0.5rem;
            display: none;
            z-index: 1100;
        }

        .dropdown.open .dropdown-menu {
            display: block;
        }

        .dropdown-menu li button {
            width: 100%;
            text-align: left;
            display: block;
            padding: 0.65rem 0.9rem;
            color: #bae6fd;
            border-radius: 6px;
            font-size: 0.92rem;
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        .dropdown-menu li button span {
            display: block;
            font-size: 0.76rem;
            color: #7dd3fc;
            opacity: 0.8;
        }

        .dropdown-menu li button:hover,
        .dropdown-menu li button.active {
            background-color: rgba(2, 132, 199, 0.3);
            color: #ffffff;
        }

        /* Section Management */
        .page-section {
            display: none;
        }

        .page-section.active-section {
            display: block;
            animation: fadeIn 0.25s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Main Content */
        main {
            max-width: 1100px;
            width: 92%;
            margin: 2.75rem auto;
            flex: 1;
        }

        .hero {
            background: linear-gradient(135deg, rgba(12, 36, 66, 0.84) 0%, rgba(7, 22, 46, 0.92) 100%);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 14px;
            padding: 3rem 2.5rem;
            margin-bottom: 3rem;
            box-shadow: 0 12px 32px rgba(2, 6, 23, 0.65);
            backdrop-filter: blur(6px);
        }

        .badge {
            display: inline-block;
            background-color: rgba(56, 189, 248, 0.15);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.4);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            margin-bottom: 1rem;
        }

        .badge-iris {
            background-color: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            border-color: rgba(16, 185, 129, 0.45);
        }

        h1 {
            font-size: 2.45rem;
            color: #ffffff;
            margin-bottom: 1rem;
            line-height: 1.2;
        }

        h2 {
            font-size: 1.6rem;
            color: #f0f9ff;
            margin: 2.5rem 0 1rem;
            border-left: 4px solid #38bdf8;
            padding-left: 0.85rem;
        }

        h3 {
            font-size: 1.15rem;
            color: #ffffff;
            margin-bottom: 0.5rem;
        }

        p {
            margin-bottom: 1rem;
            color: #cbd5e1;
            font-size: 1.01rem;
        }

        /* Buttons */
        .btn-group {
            display: flex;
            gap: 0.9rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%);
            color: #ffffff;
            padding: 0.72rem 1.35rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.94rem;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(14, 165, 233, 0.35);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
        }

        .btn-outline {
            background: rgba(14, 116, 144, 0.2);
            border: 1px solid rgba(56, 189, 248, 0.45);
            color: #e0f2fe;
            box-shadow: none;
        }

        .btn-outline:hover {
            background: rgba(2, 132, 199, 0.35);
            border-color: #38bdf8;
            color: #ffffff;
        }
