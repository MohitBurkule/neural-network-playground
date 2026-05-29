// Minimal d3 stub for unit tests (dataset.ts uses d3.scaleLinear)
const scaleLinear = () => {
  let _domain = [0, 1];
  let _range = [0, 1];
  let _clamp = false;

  function scale(value) {
    const [d0, d1] = _domain;
    const [r0, r1] = _range;
    let t = (value - d0) / (d1 - d0);
    if (_clamp) t = Math.min(1, Math.max(0, t));
    return r0 + t * (r1 - r0);
  }

  scale.domain = function(d) { if (d) { _domain = d; return scale; } return _domain; };
  scale.range  = function(r) { if (r) { _range  = r; return scale; } return _range; };
  scale.clamp  = function(c) { if (c !== undefined) { _clamp = c; return scale; } return _clamp; };

  return scale;
};

module.exports = { scaleLinear };
