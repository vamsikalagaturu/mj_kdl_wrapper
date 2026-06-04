(function () {
  "use strict";

  var storageKey = "mjk-doxygen-theme";
  var kdlMemberLinks = {
    ChainDynParam: {
      JntToCoriolis: "kdl/classKDL_1_1ChainDynParam.html#a1b6ce4f603c84038cffcc6a0800a1ae9",
      JntToGravity: "kdl/classKDL_1_1ChainDynParam.html#a229a4fa1a7be811f0a067f304777187e",
      JntToMass: "kdl/classKDL_1_1ChainDynParam.html#ae4e0adf9ff6642cbc9dec9e8b54707bc",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainDynParam.html#a7e404b5b7ea2b2182a2c341b1125edc5",
    },
    ChainFkSolverPos_recursive: {
      JntToCart: "kdl/classKDL_1_1ChainFkSolverPos__recursive.html#a9e7da71538939cdbfa8a6defe0fbc8b4",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainFkSolverPos__recursive.html#add89f7d75ae7c0c971bb958dda01643c",
    },
    ChainFkSolverVel_recursive: {
      JntToCart: "kdl/classKDL_1_1ChainFkSolverVel__recursive.html#afe0d7356e1e03ffafbbe098b115817ac",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainFkSolverVel__recursive.html#a7c366eee0c84a9c7972eb0988cfacae2",
    },
    ChainIdSolver_RNE: {
      CartToJnt: "kdl/classKDL_1_1ChainIdSolver__RNE.html#a517c0314490c32848e2b8e29ecac997e",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainIdSolver__RNE.html#a4d76fbb39d792c4f968c6685e74ad436",
    },
    ChainHdSolver_Vereshchagin: {
      CartToJnt: "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a6d48fd72f1e3fa117dba444edb0d5653",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a32274d3cc16e5e92f3d17eb8510bbaea",
    },
    ChainHdSolver_Vereshchagin_Fixed_Joint: {
      CartToJnt: "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a6d48fd72f1e3fa117dba444edb0d5653",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a32274d3cc16e5e92f3d17eb8510bbaea",
    },
    ChainIdSolver_Vereshchagin: {
      CartToJnt: "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a6d48fd72f1e3fa117dba444edb0d5653",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainHdSolver__Vereshchagin.html#a32274d3cc16e5e92f3d17eb8510bbaea",
    },
    ChainIkSolverPos_LMA: {
      CartToJnt: "kdl/classKDL_1_1ChainIkSolverPos__LMA.html#a95cbaf66c208d36148a98d0b68f5db00",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainIkSolverPos__LMA.html#a068a7af864c449f5053e3ba528c849fb",
    },
    ChainIkSolverPos_NR_JL: {
      CartToJnt: "kdl/classKDL_1_1ChainIkSolverPos__NR__JL.html#a3aeeca64ab0233666b8304c99cfb5365",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainIkSolverPos__NR__JL.html#a4ef6c0744665fa0871bdbfe98edec25e",
    },
    ChainIkSolverVel_pinv: {
      CartToJnt: "kdl/classKDL_1_1ChainIkSolverVel__pinv.html#ad5cb83f89dc128ddd79c317fa597de99",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainIkSolverVel__pinv.html#a6c47977458159f9cc67325978cd7abed",
    },
    ChainIkSolverVel_wdls: {
      CartToJnt: "kdl/classKDL_1_1ChainIkSolverVel__wdls.html#aee1de1420b70666622da81477a94bcf5",
      updateInternalDataStructures:
        "kdl/classKDL_1_1ChainIkSolverVel__wdls.html#a3e14f470112d10981ca090308866ec22",
    },
  };
  var bareKdlMemberLinks = {
    JntToCoriolis: kdlMemberLinks.ChainDynParam.JntToCoriolis,
    JntToGravity: kdlMemberLinks.ChainDynParam.JntToGravity,
    JntToMass: kdlMemberLinks.ChainDynParam.JntToMass,
  };
  var commonKdlVariables = {
    achd: "ChainHdSolver_Vereshchagin_Fixed_Joint",
    dyn: "ChainDynParam",
    fk: "ChainFkSolverPos_recursive",
    fk_pos: "ChainFkSolverPos_recursive",
    fk_solver: "ChainFkSolverPos_recursive",
    fk_vel: "ChainFkSolverVel_recursive",
    ik_lma: "ChainIkSolverPos_LMA",
    ik_nr: "ChainIkSolverPos_NR_JL",
    ik_pos: "ChainIkSolverPos_NR_JL",
    ik_vel: "ChainIkSolverVel_pinv",
    rnea: "ChainIdSolver_RNE",
  };
  var pythonNamespaceHref = "namespacemj__kdl__wrapper_1_1__mj__kdl__wrapper.html";
  var pythonObjectLinks = {
    AttachKind: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1AttachKind.html",
    AttachTarget: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1AttachTarget.html",
    AttachmentSpec: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1AttachmentSpec.html",
    CameraSpec: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1CameraSpec.html",
    Condim: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1Condim.html",
    CtrlMode: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1CtrlMode.html",
    Env: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1Env.html",
    LogLevel: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1LogLevel.html",
    ResetInfo: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1ResetInfo.html",
    ResetOptions: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1ResetOptions.html",
    Robot: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1Robot.html",
    RobotSpec: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1RobotSpec.html",
    Scene: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1Scene.html",
    SceneObject: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1SceneObject.html",
    SceneSpec: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1SceneSpec.html",
    Shape: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1Shape.html",
    SimulateViewer: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1SimulateViewer.html",
    ToolFrameSpec: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1ToolFrameSpec.html",
    VideoRecorder: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1VideoRecorder.html",
    VideoResolution: "classmj__kdl__wrapper_1_1__mj__kdl__wrapper_1_1VideoResolution.html",
  };
  var commonPythonVariables = {
    env: "Env",
    recorder: "VideoRecorder",
    robot: "Robot",
    robot_spec: "RobotSpec",
    scene: "Scene",
    spec: "SceneSpec",
    viewer: "SimulateViewer",
  };
  var pythonMethodNames = {
    add_object: true,
    add_robot: true,
    body_frame: true,
    build: true,
    close: true,
    fk_frame: true,
    from_scene: true,
    from_spec: true,
    kdl_chain: true,
    remove_object: true,
    reset: true,
    save_xml: true,
    site_frame: true,
    start: true,
    step: true,
    stop: true,
    update: true,
  };

  function systemPrefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function currentTheme() {
    var saved = window.localStorage ? window.localStorage.getItem(storageKey) : null;
    if (saved === "light" || saved === "dark") {
      return saved;
    }
    return systemPrefersDark() ? "dark" : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    var button = document.getElementById("mjk-theme-toggle");
    if (button) {
      button.textContent = theme === "dark" ? "Light" : "Dark";
      button.setAttribute("aria-label", "Switch to " + (theme === "dark" ? "light" : "dark") + " mode");
    }
  }

  function shouldSkipCodeTextNode(node) {
    var parent = node.parentElement;
    if (!parent) {
      return true;
    }
    return Boolean(
      parent.closest(
        "a, span.keyword, span.keywordtype, span.keywordflow, span.comment, " +
          "span.preprocessor, span.stringliteral, span.charliteral, span.lineno, " +
          ".mjk-hl-type, .mjk-hl-pyobject, .mjk-hl-pyfunction"
      )
    );
  }

  function directFragmentLineText(fragment) {
    var lines = [];

    fragment.querySelectorAll("div.line").forEach(function (line) {
      if (line.closest(".fragment") === fragment) {
        lines.push(line.textContent);
      }
    });

    return lines.join("\n");
  }

  function inferKdlVariables(fragment) {
    var variables = Object.create(null);
    var text = directFragmentLineText(fragment);
    var declarationPattern = /\bKDL::([A-Za-z_][A-Za-z0-9_]*)\s+(?:[*&]\s*)?([A-Za-z_][A-Za-z0-9_]*)\b/g;
    var match;

    Object.keys(commonKdlVariables).forEach(function (name) {
      variables[name] = commonKdlVariables[name];
    });

    while ((match = declarationPattern.exec(text)) !== null) {
      if (Object.prototype.hasOwnProperty.call(kdlMemberLinks, match[1])) {
        variables[match[2]] = match[1];
      }
    }

    return variables;
  }

  function hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj, key);
  }

  function unresolvedKdlMemberHref(variables, variableName, memberName) {
    if (!hasOwn(variables, variableName)) {
      return null;
    }

    var classLinks = kdlMemberLinks[variables[variableName]];
    return classLinks && hasOwn(classLinks, memberName) ? classLinks[memberName] : null;
  }

  function createKdlFunctionLink(name, href) {
    var link = document.createElement("a");
    link.className = "code hl_functionRef";
    link.href = href;
    link.textContent = name;
    return link;
  }

  function createPythonClassLink(name, href) {
    var link = document.createElement("a");
    link.className = "code hl_classRef";
    link.href = href;
    link.textContent = name;
    return link;
  }

  function createPythonFunctionLink(name, href) {
    var link = document.createElement("a");
    link.className = "code hl_functionRef";
    link.href = href;
    link.textContent = name;
    return link;
  }

  function pythonObjectHref(className) {
    return hasOwn(pythonObjectLinks, className) ? pythonObjectLinks[className] : null;
  }

  function inferPythonVariables(fragment) {
    var variables = Object.create(null);
    var text = directFragmentLineText(fragment);
    var assignmentPattern =
      /\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:mjk|mj_kdl_wrapper)\.([A-Z][A-Za-z0-9_]*)\b/g;
    var match;

    Object.keys(commonPythonVariables).forEach(function (name) {
      variables[name] = commonPythonVariables[name];
    });

    while ((match = assignmentPattern.exec(text)) !== null) {
      if (hasOwn(pythonObjectLinks, match[2])) {
        variables[match[1]] = match[2];
      }
    }

    return variables;
  }

  function wrapPythonApiNode(node, pattern) {
    var text = node.nodeValue;
    var last = 0;
    var fragment = document.createDocumentFragment();
    var match;

    pattern.lastIndex = 0;
    while ((match = pattern.exec(text)) !== null) {
      var alias = match[1] || match[4];
      var className = match[2];
      var memberName = match[3];
      var functionName = match[5];
      var classHref = className ? pythonObjectHref(className) : null;

      if (className && !classHref) {
        continue;
      }

      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }

      if (className) {
        fragment.appendChild(createPythonClassLink(alias + "." + className, classHref));
        if (memberName) {
          fragment.appendChild(document.createTextNode("."));
          fragment.appendChild(
            pythonMethodNames[memberName]
              ? createPythonFunctionLink(memberName, classHref)
              : createPythonClassLink(memberName, classHref)
          );
        }
      } else {
        fragment.appendChild(createPythonFunctionLink(alias + "." + functionName, pythonNamespaceHref));
      }

      last = match.index + match[0].length;
    }

    if (last === 0) {
      return;
    }

    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
    node.parentNode.replaceChild(fragment, node);
  }

  function wrapPythonVariableMemberNode(node, variables, pattern) {
    var text = node.nodeValue;
    var last = 0;
    var fragment = document.createDocumentFragment();
    var match;

    pattern.lastIndex = 0;
    while ((match = pattern.exec(text)) !== null) {
      if (!hasOwn(variables, match[1]) || !pythonMethodNames[match[2]]) {
        continue;
      }

      var href = pythonObjectHref(variables[match[1]]);
      if (!href) {
        continue;
      }

      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      fragment.appendChild(document.createTextNode(match[1] + "."));
      fragment.appendChild(createPythonFunctionLink(match[2], href));
      last = match.index + match[0].length;
    }

    if (last === 0) {
      return;
    }

    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
    node.parentNode.replaceChild(fragment, node);
  }

  function linkPythonApiReferences() {
    var apiPattern =
      /\b(mjk|mj_kdl_wrapper)\.([A-Z][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?\b|\b(mjk|mj_kdl_wrapper)\.([a-z_][A-Za-z0-9_]*)\b/g;
    var methodPattern =
      /\b([A-Za-z_][A-Za-z0-9_]*)\.(add_object|add_robot|body_frame|build|close|fk_frame|from_scene|from_spec|kdl_chain|remove_object|reset|save_xml|site_frame|start|step|stop|update)\b/g;

    document.querySelectorAll(".fragment").forEach(function (fragment) {
      var variables = inferPythonVariables(fragment);

      fragment.querySelectorAll("div.line").forEach(function (line) {
        if (line.closest(".fragment") !== fragment) {
          return;
        }

        var nodes = [];
        var walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, {
          acceptNode: function (node) {
            if (shouldSkipCodeTextNode(node)) {
              return NodeFilter.FILTER_REJECT;
            }
            apiPattern.lastIndex = 0;
            return apiPattern.test(node.nodeValue)
              ? NodeFilter.FILTER_ACCEPT
              : NodeFilter.FILTER_REJECT;
          },
        });

        while (walker.nextNode()) {
          nodes.push(walker.currentNode);
        }

        nodes.forEach(function (node) {
          wrapPythonApiNode(node, apiPattern);
        });

        nodes = [];
        walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, {
          acceptNode: function (node) {
            if (shouldSkipCodeTextNode(node)) {
              return NodeFilter.FILTER_REJECT;
            }
            methodPattern.lastIndex = 0;
            return methodPattern.test(node.nodeValue)
              ? NodeFilter.FILTER_ACCEPT
              : NodeFilter.FILTER_REJECT;
          },
        });

        while (walker.nextNode()) {
          nodes.push(walker.currentNode);
        }

        nodes.forEach(function (node) {
          wrapPythonVariableMemberNode(node, variables, methodPattern);
        });
      });
    });
  }

  function wrapUnresolvedKdlMemberNode(node, variables, pattern) {
    var text = node.nodeValue;
    var last = 0;
    var fragment = document.createDocumentFragment();
    var match;

    pattern.lastIndex = 0;
    while ((match = pattern.exec(text)) !== null) {
      var href = unresolvedKdlMemberHref(variables, match[1], match[2]);
      if (!href) {
        continue;
      }

      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      fragment.appendChild(document.createTextNode(match[1] + "."));
      fragment.appendChild(createKdlFunctionLink(match[2], href));
      last = match.index + match[0].length;
    }

    if (last === 0) {
      return;
    }

    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
    node.parentNode.replaceChild(fragment, node);
  }

  function wrapBareKdlMemberNode(node, pattern) {
    var text = node.nodeValue;
    var last = 0;
    var fragment = document.createDocumentFragment();
    var match;

    pattern.lastIndex = 0;
    while ((match = pattern.exec(text)) !== null) {
      var href = bareKdlMemberLinks[match[1]];
      if (!href) {
        continue;
      }

      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      fragment.appendChild(createKdlFunctionLink(match[1], href));
      last = match.index + match[0].length;
    }

    if (last === 0) {
      return;
    }

    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
    node.parentNode.replaceChild(fragment, node);
  }

  function linkUnresolvedKdlMemberCalls() {
    var memberCallPattern =
      /\b([A-Za-z_][A-Za-z0-9_]*)\.(JntToGravity|JntToCoriolis|JntToMass|JntToCart|CartToJnt|updateInternalDataStructures)\b/g;
    var bareMemberPattern = /\b(JntToGravity|JntToCoriolis|JntToMass)\b/g;

    document.querySelectorAll(".fragment").forEach(function (fragment) {
      var variables = inferKdlVariables(fragment);

      fragment.querySelectorAll("div.line").forEach(function (line) {
        if (line.closest(".fragment") !== fragment) {
          return;
        }

        var memberNodes = [];
        var bareNodes = [];
        var walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, {
          acceptNode: function (node) {
            if (shouldSkipCodeTextNode(node)) {
              return NodeFilter.FILTER_REJECT;
            }
            memberCallPattern.lastIndex = 0;
            bareMemberPattern.lastIndex = 0;
            if (memberCallPattern.test(node.nodeValue)) {
              return NodeFilter.FILTER_ACCEPT;
            }
            return bareMemberPattern.test(node.nodeValue)
              ? NodeFilter.FILTER_ACCEPT
              : NodeFilter.FILTER_REJECT;
          },
        });

        while (walker.nextNode()) {
          memberNodes.push(walker.currentNode);
        }

        memberNodes.forEach(function (node) {
          wrapUnresolvedKdlMemberNode(node, variables, memberCallPattern);
        });

        walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, {
          acceptNode: function (node) {
            if (shouldSkipCodeTextNode(node)) {
              return NodeFilter.FILTER_REJECT;
            }
            bareMemberPattern.lastIndex = 0;
            return bareMemberPattern.test(node.nodeValue)
              ? NodeFilter.FILTER_ACCEPT
              : NodeFilter.FILTER_REJECT;
          },
        });

        while (walker.nextNode()) {
          bareNodes.push(walker.currentNode);
        }

        bareNodes.forEach(function (node) {
          wrapBareKdlMemberNode(node, bareMemberPattern);
        });
      });
    });
  }

  function wrapUnresolvedTypeNode(node, pattern) {
    var text = node.nodeValue;
    var last = 0;
    var fragment = document.createDocumentFragment();
    var match;

    pattern.lastIndex = 0;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) {
        fragment.appendChild(document.createTextNode(text.slice(last, match.index)));
      }

      var span = document.createElement("span");
      span.className = "mjk-hl-type";
      span.textContent = match[0];
      fragment.appendChild(span);
      last = match.index + match[0].length;
    }

    if (last === 0) {
      return;
    }

    if (last < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(last)));
    }
    node.parentNode.replaceChild(fragment, node);
  }

  function highlightUnresolvedCodeTypes() {
    var unresolvedTypePattern =
      /\b(?:KDL|std)::[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*\b|\bmjk\.[A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b|\bPyKDL\.[A-Z][A-Za-z0-9_]*\b|\bmj(?:Model|Data)\b/g;

    document.querySelectorAll(".fragment div.line").forEach(function (line) {
      var nodes = [];
      var walker = document.createTreeWalker(line, NodeFilter.SHOW_TEXT, {
        acceptNode: function (node) {
          if (shouldSkipCodeTextNode(node)) {
            return NodeFilter.FILTER_REJECT;
          }
          unresolvedTypePattern.lastIndex = 0;
          return unresolvedTypePattern.test(node.nodeValue)
            ? NodeFilter.FILTER_ACCEPT
            : NodeFilter.FILTER_REJECT;
        },
      });

      while (walker.nextNode()) {
        nodes.push(walker.currentNode);
      }
      nodes.forEach(function (node) {
        wrapUnresolvedTypeNode(node, unresolvedTypePattern);
      });
    });
  }

  function init() {
    var button = document.getElementById("mjk-theme-toggle");
    if (button) {
      button.addEventListener("click", function () {
        var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        if (window.localStorage) {
          window.localStorage.setItem(storageKey, next);
        }
        applyTheme(next);
      });
    }
    linkPythonApiReferences();
    linkUnresolvedKdlMemberCalls();
    highlightUnresolvedCodeTypes();
  }

  applyTheme(currentTheme());

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
