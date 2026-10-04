## 🎬 Demo

<div align="center">

<img src="media/gif-3body2.gif" alt="3D animation of a long-lived three-body system" width="780">

<sub><b>A surviving system replayed in the visualiser:</b> three bodies sized by mass, with fading trails.</sub>

</div>

<br>

### 🖥️ The search in action

The search runs thousands of random systems back to back. Most end in a quick collision or ejection, and the rare long-lived ones are flagged automatically.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="media/3bodypics1.png" alt="High priority case found in the terminal">
      <br><sub><b>Rare find:</b> case 54 survives all 500,000 steps and is ranked <i>Very High</i>.</sub>
    </td>
    <td align="center" width="50%">
      <img src="media/3bodypics2.png" alt="Search progress showing collisions and ejections">
      <br><sub><b>Typical progress:</b> most cases end in collision, a few in ejection.</sub>
    </td>
  </tr>
</table>

### 🔭 The visualiser

Replay any case with `python visualize_case.py 54`. It shows the 3D motion, the pair-distance graph and the run statistics together.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="media/3bodypics3.png" alt="Visualiser at t = 4.49">
      <br><sub><b>t = 4.49:</b> the heavy body and the middle body form a tight pair.</sub>
    </td>
    <td align="center" width="50%">
      <img src="media/3bodypics4.png" alt="Visualiser at t = 12.21">
      <br><sub><b>t = 12.21:</b> the light body is flung outward on a long loop.</sub>
    </td>
  </tr>
</table>

> 💡 The <b>top-right graph</b> is the quickest way to read a system: the pair distances must stay above the dashed red <i>collision distance</i> line for the system to survive.

---
