import json
from typing import Dict, Any, Optional
from kg_engine import KnowledgeGraphEngine

def generate_interactive_graph_html(
    kg_engine: KnowledgeGraphEngine,
    height: int = 650,
    background_color: str = "#0B0F19"
) -> str:
    """Generate a self-contained, responsive, high-performance HTML/JS string

    rendering an interactive Vis.js Network knowledge graph with dark aesthetic,
    node search, click inspector, physics controls, and cross-paper bridge highlights.
    """
    graph_data = kg_engine.to_vis_network_data()
    bridges = kg_engine.get_cross_paper_bridges()
    
    nodes_json = json.dumps(graph_data["nodes"])
    edges_json = json.dumps(graph_data["edges"])
    bridges_json = json.dumps(bridges)
    
    # Paper metadata dictionary for rich inspector cards
    papers_meta = {}
    for doc_id, meta in kg_engine.documents_metadata.items():
        papers_meta[doc_id] = {
            "title": meta.get("title", doc_id),
            "source": meta.get("source", "Manual"),
            "url": meta.get("url", ""),
            "authors": meta.get("authors", []),
            "year": meta.get("published_date", "")
        }
    papers_meta_json = json.dumps(papers_meta)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive Knowledge Graph Explorer</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    body {{
      background-color: {background_color};
      color: #E2E8F0;
      overflow: hidden;
      user-select: none;
    }}
    #graph-container {{
      position: relative;
      width: 100%;
      height: {height}px;
      background: radial-gradient(circle at 50% 50%, #151D30 0%, #0B0F19 100%);
      border: 1px solid rgba(0, 210, 255, 0.2);
      border-radius: 12px;
      overflow: hidden;
    }}
    #mynetwork {{
      width: 100%;
      height: 100%;
    }}
    /* Top HUD Navigation Bar */
    .hud-bar {{
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      z-index: 10;
      pointer-events: none;
    }}
    .hud-group {{
      display: flex;
      align-items: center;
      gap: 8px;
      pointer-events: auto;
    }}
    .hud-pill {{
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(79, 209, 197, 0.3);
      border-radius: 8px;
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 500;
      color: #CBD5E0;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .hud-pill b {{
      color: #00D2FF;
      font-weight: 700;
    }}
    .hud-btn {{
      background: rgba(26, 32, 44, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 210, 255, 0.35);
      color: #E2E8F0;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .hud-btn:hover {{
      background: rgba(0, 210, 255, 0.25);
      border-color: #00D2FF;
      color: #FFFFFF;
      transform: translateY(-1px);
    }}
    .hud-input {{
      background: rgba(17, 24, 39, 0.9);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(0, 210, 255, 0.3);
      color: #FFFFFF;
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 12px;
      outline: none;
      width: 170px;
      transition: border 0.2s ease;
    }}
    .hud-input:focus {{
      border-color: #00D2FF;
      box-shadow: 0 0 8px rgba(0, 210, 255, 0.4);
    }}
    /* Inspector Card Drawer */
    #inspector {{
      position: absolute;
      bottom: 14px;
      right: 14px;
      width: 320px;
      max-height: 380px;
      background: rgba(17, 24, 39, 0.92);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(0, 210, 255, 0.4);
      border-radius: 12px;
      padding: 16px;
      overflow-y: auto;
      z-index: 20;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      display: none;
      animation: fadeIn 0.25s ease;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(10px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .inspector-header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 8px;
    }}
    .inspector-title {{
      font-size: 14px;
      font-weight: 700;
      color: #00E5FF;
      line-height: 1.3;
    }}
    .inspector-close {{
      cursor: pointer;
      color: #A0AEC0;
      font-size: 16px;
      line-height: 1;
      padding: 2px 6px;
    }}
    .inspector-close:hover {{
      color: #FFFFFF;
    }}
    .inspector-section {{
      margin-top: 10px;
    }}
    .inspector-section-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #718096;
      margin-bottom: 6px;
      font-weight: 700;
    }}
    .inspector-badge {{
      display: inline-block;
      background: rgba(0, 210, 255, 0.15);
      color: #00D2FF;
      border: 1px solid rgba(0, 210, 255, 0.3);
      border-radius: 4px;
      padding: 2px 6px;
      font-size: 11px;
      margin: 2px 4px 2px 0;
    }}
    .inspector-relation {{
      background: rgba(255, 255, 255, 0.05);
      border-left: 3px solid #10B981;
      padding: 6px 8px;
      border-radius: 4px;
      font-size: 11px;
      margin-bottom: 5px;
      line-height: 1.4;
    }}
    .legend-bar {{
      position: absolute;
      bottom: 14px;
      left: 14px;
      background: rgba(17, 24, 39, 0.85);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 8px;
      padding: 8px 12px;
      display: flex;
      gap: 12px;
      font-size: 11px;
      color: #A0AEC0;
      z-index: 10;
      pointer-events: auto;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .empty-state {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      color: #718096;
    }}
    .empty-state h3 {{
      color: #00D2FF;
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <div id="graph-container">
    <div class="hud-bar">
      <div class="hud-group">
        <div class="hud-pill">Nodes: <b id="node-count">0</b></div>
        <div class="hud-pill">Relations: <b id="edge-count">0</b></div>
        <div class="hud-pill">Cross-Paper Hubs: <b id="hub-count">0</b></div>
      </div>
      <div class="hud-group">
        <input type="text" id="search-input" class="hud-input" placeholder="Search concept..." onkeyup="searchNode(event)">
        <button class="hud-btn" onclick="focusSelectedOrSearched()">🔍 Find</button>
        <button class="hud-btn" onclick="resetZoom()">🎯 Fit View</button>
        <button class="hud-btn" id="physics-toggle-btn" onclick="togglePhysics()">⏸️ Freeze</button>
      </div>
    </div>

    <div id="mynetwork"></div>

    <div class="legend-bar">
      <div class="legend-item">
        <div class="legend-dot" style="background: #00E5FF; box-shadow: 0 0 6px #00E5FF;"></div>
        <span>Cross-Paper Hub</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #F59E0B;"></div>
        <span>Scientific Paper</span>
      </div>
      <div class="legend-item">
        <div class="legend-dot" style="background: #4FD1C5;"></div>
        <span>Biological / Physical Entity</span>
      </div>
    </div>

    <!-- Click Inspector Drawer -->
    <div id="inspector">
      <div class="inspector-header">
        <div class="inspector-title" id="inspector-name">Concept Name</div>
        <div class="inspector-close" onclick="closeInspector()">&times;</div>
      </div>
      <div class="inspector-section">
        <div class="inspector-section-title">Cited in Research Papers:</div>
        <div id="inspector-sources"></div>
      </div>
      <div class="inspector-section">
        <div class="inspector-section-title">Knowledge Graph Relations:</div>
        <div id="inspector-relations"></div>
      </div>
    </div>

    <div id="empty-notice" class="empty-state" style="display: none;">
      <h3>Knowledge Graph is Empty</h3>
      <p>Search arXiv / PubMed or ingest research papers to build your visual interactive map.</p>
    </div>
  </div>

  <script type="text/javascript">
    const rawNodes = {nodes_json};
    const rawEdges = {edges_json};
    const bridges = {bridges_json};
    const papersMeta = {papers_meta_json};

    document.getElementById("node-count").innerText = rawNodes.length;
    document.getElementById("edge-count").innerText = rawEdges.length;
    document.getElementById("hub-count").innerText = bridges.length;

    if (rawNodes.length === 0) {{
      document.getElementById("empty-notice").style.display = "block";
    }}

    const container = document.getElementById("mynetwork");
    const nodes = new vis.DataSet(rawNodes);
    const edges = new vis.DataSet(rawEdges);

    const data = {{
      nodes: nodes,
      edges: edges
    }};

    let physicsEnabled = true;

    const options = {{
      nodes: {{
        borderWidth: 1.5,
        borderWidthSelected: 3,
        shadow: true,
        font: {{
          color: '#F7FAFC',
          size: 12,
          face: 'Inter, system-ui, sans-serif'
        }}
      }},
      edges: {{
        width: 1.4,
        hoverWidth: 2.4,
        selectionWidth: 2.8,
        arrows: {{
          to: {{ enabled: true, scaleFactor: 0.7 }}
        }},
        smooth: {{
          type: 'continuous',
          roundness: 0.2
        }}
      }},
      physics: {{
        stabilization: {{ iterations: 120 }},
        barnesHut: {{
          gravitationalConstant: -3500,
          centralGravity: 0.35,
          springLength: 95,
          springConstant: 0.04,
          damping: 0.09
        }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 150,
        hideEdgesOnDrag: false,
        navigationButtons: false,
        keyboard: false
      }}
    }};

    const network = new vis.Network(container, data, options);

    // Event: Click on node to inspect connections
    network.on("click", function(params) {{
      if (params.nodes.length > 0) {{
        const nodeId = params.nodes[0];
        inspectNode(nodeId);
      }} else {{
        closeInspector();
      }}
    }});

    function inspectNode(nodeId) {{
      const node = nodes.get(nodeId);
      if (!node) return;

      document.getElementById("inspector-name").innerText = nodeId;
      
      // Find connected edges
      const connectedEdges = network.getConnectedEdges(nodeId);
      const relationsDiv = document.getElementById("inspector-relations");
      relationsDiv.innerHTML = "";

      let foundRelations = 0;
      connectedEdges.forEach(edgeId => {{
        const edge = edges.get(edgeId);
        if (edge) {{
          foundRelations++;
          const isOutgoing = edge.from === nodeId;
          const target = isOutgoing ? edge.to : edge.from;
          const arrow = isOutgoing ? "&rarr;" : "&larr;";
          const relBox = document.createElement("div");
          relBox.className = "inspector-relation";
          relBox.innerHTML = `<span>${{nodeId}}</span> <b>${{arrow}} [${{edge.label}}] ${{arrow}}</b> <span>${{target}}</span>`;
          relationsDiv.appendChild(relBox);
        }}
      }});

      if (foundRelations === 0) {{
        relationsDiv.innerHTML = "<small style='color: #718096;'>No outgoing relations recorded.</small>";
      }}

      // Find sources
      const sourcesDiv = document.getElementById("inspector-sources");
      sourcesDiv.innerHTML = "";
      
      // Check node group and bridge data
      const bridgeMatch = bridges.find(b => b.concept.toLowerCase() === nodeId.toLowerCase());
      const sourceList = bridgeMatch ? bridgeMatch.connected_papers : (node.group ? [node.group] : []);

      if (sourceList.length > 0) {{
        sourceList.forEach(src => {{
          const meta = papersMeta[src];
          const badge = document.createElement("div");
          badge.className = "inspector-badge";
          if (meta && meta.url) {{
            badge.innerHTML = `<a href="${{meta.url}}" target="_blank" style="color: #00D2FF; text-decoration: none;">📄 ${{src}} &nearr;</a>`;
          }} else {{
            badge.innerText = `📄 ${{src}}`;
          }}
          sourcesDiv.appendChild(badge);
        }});
      }} else {{
        sourcesDiv.innerHTML = "<span class='inspector-badge'>Research Graph Entity</span>";
      }}

      document.getElementById("inspector").style.display = "block";
    }}

    function closeInspector() {{
      document.getElementById("inspector").style.display = "none";
    }}

    function searchNode(event) {{
      if (event.key === "Enter") {{
        focusSelectedOrSearched();
      }}
    }}

    function focusSelectedOrSearched() {{
      const query = document.getElementById("search-input").value.trim().toLowerCase();
      if (!query) return;

      const allNodes = nodes.get();
      const matched = allNodes.find(n => n.id.toLowerCase().includes(query) || (n.label && n.label.toLowerCase().includes(query)));
      
      if (matched) {{
        network.focus(matched.id, {{
          scale: 1.3,
          animation: {{
            duration: 800,
            easingFunction: 'easeInOutQuad'
          }}
        }});
        network.selectNodes([matched.id]);
        inspectNode(matched.id);
      }} else {{
        alert("Concept not found in Knowledge Graph: " + query);
      }}
    }}

    function resetZoom() {{
      network.fit({{
        animation: {{
          duration: 600,
          easingFunction: 'easeInOutQuad'
        }}
      }});
    }}

    function togglePhysics() {{
      physicsEnabled = !physicsEnabled;
      network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
      const btn = document.getElementById("physics-toggle-btn");
      if (physicsEnabled) {{
        btn.innerHTML = "⏸️ Freeze";
      }} else {{
        btn.innerHTML = "▶️ Unfreeze";
      }}
    }}
  </script>
</body>
</html>
"""
    return html

