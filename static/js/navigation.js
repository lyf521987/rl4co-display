/**
 * 统一导航栏组件和页面跳转逻辑
 * RL4CO Display - 山西大学
 * 优化版本：减少卡顿，提升响应速度
 */

// ============================================
// 0. 性能监测工具（可选）
// ============================================

const perfMetrics = {
    navigationStart: 0,
    navigationEnd: 0
};

// ============================================
// 1. 导航栏高亮管理（优化版）
// ============================================

/**
 * 根据当前URL设置导航栏激活状态 - 使用事件委托优化
 */
function setActiveNavLink() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    // 批量移除和添加active类
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        const isActive = href === currentPath || 
                        (currentPath === '/' && href === '#home') ||
                        (currentPath === '/home' && href === '#home');
        
        link.classList.toggle('active', isActive);
    });
}

/**
 * 监听锚点变化，更新导航高亮
 */
function handleHashChange() {
    const hash = window.location.hash;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        const isActive = href === hash || (hash === '' && href === '#home');
        link.classList.toggle('active', isActive);
    });
}

// ============================================
// 2. 平滑滚动效果（优化版）
// ============================================

/**
 * 为所有锚点链接添加平滑滚动 - 使用事件委托
 */
function initSmoothScroll() {
    // 使用事件委托而不是为每个链接添加监听
    document.addEventListener('click', function(e) {
        const link = e.target.closest('a[href^="#"]');
        if (!link) return;
        
        const href = link.getAttribute('href');
        if (href === '#') return;
        
        const targetId = href.substring(1);
        const targetElement = document.getElementById(targetId);
        
        if (targetElement) {
            e.preventDefault();
            
            const headerHeight = document.querySelector('.header')?.offsetHeight || 0;
            const targetPosition = targetElement.offsetTop - headerHeight - 20;
            
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
            
            history.pushState(null, null, href);
            handleHashChange();
        }
    }, true); // 使用捕获阶段以提高性能
}

// ============================================
// 3. 移动端导航菜单（优化版）
// ============================================

/**
 * 初始化移动端汉堡菜单 - 只初始化一次
 */
function initMobileMenu() {
    const nav = document.querySelector('.nav');
    const navLinks = document.querySelector('.nav-links');
    
    if (!nav || !navLinks) return;
    
    if (document.querySelector('.mobile-menu-btn')) return; // 已初始化
    
    const menuBtn = document.createElement('button');
    menuBtn.className = 'mobile-menu-btn';
    menuBtn.innerHTML = '☰';
    menuBtn.setAttribute('aria-label', '菜单');
    
    nav.appendChild(menuBtn);
    
    // 切换菜单显示 - 优化版
    menuBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        navLinks.classList.toggle('mobile-show');
        this.classList.toggle('active');
        this.innerHTML = this.classList.contains('active') ? '✕' : '☰';
    });
    
    // 点击菜单项后关闭菜单 - 使用事件委托
    navLinks.addEventListener('click', function(e) {
        if (e.target.tagName === 'A' && window.innerWidth <= 768) {
            navLinks.classList.remove('mobile-show');
            menuBtn.classList.remove('active');
            menuBtn.innerHTML = '☰';
        }
    });
    
    // 点击外部关闭菜单
    document.addEventListener('click', function(e) {
        if (!nav.contains(e.target) && navLinks.classList.contains('mobile-show')) {
            navLinks.classList.remove('mobile-show');
            menuBtn.classList.remove('active');
            menuBtn.innerHTML = '☰';
        }
    });
}

// ============================================
// 4. 返回顶部按钮（优化版）
// ============================================

/**
 * 创建并管理返回顶部按钮 - 优化的滚动监听
 */
let backToTopBtn = null;
let scrollTimeout = null;

function initBackToTop() {
    if (backToTopBtn) return; // 已初始化
    
    backToTopBtn = document.createElement('button');
    backToTopBtn.className = 'back-to-top';
    backToTopBtn.innerHTML = '↑';
    backToTopBtn.setAttribute('aria-label', '返回顶部');
    backToTopBtn.style.display = 'none';
    
    document.body.appendChild(backToTopBtn);
    
    // 使用throttle优化滚动监听
    window.addEventListener('scroll', function() {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(function() {
            const shouldShow = window.pageYOffset > 300;
            backToTopBtn.style.display = shouldShow ? 'flex' : 'none';
        }, 100);
    }, { passive: true }); // 使用passive标志提高滚动性能
    
    // 返回顶部
    backToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

// ============================================
// 5. 页面加载动画（优化版）
// ============================================

/**
 * 快速的页面加载效果 - 移除不必要的延迟
 */
function initPageLoadAnimation() {
    // 移除之前的淡入效果，减少卡顿
    // 使用CSS直接设置，而不是JavaScript
}

let pageLoading = null;

function showPageLoading() {
    if (!pageLoading) {
        pageLoading = document.createElement('div');
        pageLoading.className = 'page-loading';
        pageLoading.innerHTML = `
            <div class="loading-spinner"></div>
            <p>加载中...</p>
        `;
        document.body.appendChild(pageLoading);
    } else {
        pageLoading.style.display = 'flex';
    }
}

function hidePageLoading() {
    if (pageLoading) {
        pageLoading.style.display = 'none';
    }
}

// ============================================
// 6. 路由管理（SPA风格 - 优化版）
// ============================================

/**
 * 拦截页面链接，实现快速过渡 - 移除不必要的延迟
 */
function initRouteTransition() {
    // 使用事件委托而不是为每个链接添加监听
    document.addEventListener('click', function(e) {
        const link = e.target.closest('a[href^="/"]');
        if (!link) return;
        
        const href = link.getAttribute('href');
        
        // 排除外部链接和特殊链接
        if (href.startsWith('http') || href.includes('logout')) {
            return;
        }
        
        // 防止重复导航
        if (href === window.location.pathname) {
            e.preventDefault();
            return;
        }
        
        // 立即导航，使用浏览器原生性能
        // 移除300ms延迟，依赖浏览器的page-transition API
        // showPageLoading(); // 可选：仅在需要时显示
        
        // 继续进行默认导航（浏览器处理会更快）
    }, true);
}

// ============================================
// 7. 面包屑导航（优化版）
// ============================================

/**
 * 根据当前页面生成面包屑导航
 */
function initBreadcrumb() {
    const breadcrumbContainer = document.querySelector('.breadcrumb');
    if (!breadcrumbContainer) return;
    
    const path = window.location.pathname;
    const pathMap = {
        '/': '首页',
        '/home': '首页',
        '/benchmark': '算法对比',
        '/file_manager': '文件管理',
        '/model_info': '模型详情',
        '/profile': '我的账户'
    };
    
    let breadcrumbHTML = '<a href="/">首页</a>';
    
    if (path !== '/' && path !== '/home') {
        const currentPageName = pathMap[path] || '页面';
        breadcrumbHTML += ` <span class="separator">›</span> <span class="current">${currentPageName}</span>`;
    }
    
    breadcrumbContainer.innerHTML = breadcrumbHTML;
}

// ============================================
// 8. 初始化所有功能（优化版）
// ============================================

/**
 * 延迟初始化策略 - 只在需要时初始化
 */
function initNavigationModule() {
    // 立即执行（影响UX的功能）
    setActiveNavLink();
    
    // 页面加载完成后执行
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', completeInit);
    } else {
        completeInit();
    }
}

function completeInit() {
    // 处理hash变化
    if (window.location.pathname === '/' || window.location.pathname === '/home') {
        handleHashChange();
        window.addEventListener('hashchange', handleHashChange);
    }
    
    // 初始化交互功能
    initSmoothScroll();
    initMobileMenu();
    initBackToTop();
    initRouteTransition();
    initBreadcrumb();
    
    // 隐藏加载动画
    hidePageLoading();
}

// ============================================
// 9. 工具函数
// ============================================

/**
 * 获取当前页面名称
 */
function getCurrentPageName() {
    const path = window.location.pathname;
    const pageMap = {
        '/': '首页',
        '/home': '首页',
        '/benchmark': '算法对比',
        '/file_manager': '文件管理',
        '/model_info': '模型详情',
        '/profile': '我的账户'
    };
    return pageMap[path] || '页面';
}

/**
 * 快速安全导航 - 移除不必要延迟
 */
function navigateTo(url, showLoading = false) {
    if (showLoading) {
        showPageLoading();
    }
    window.location.href = url;
}

// ============================================
// 10. 初始化
// ============================================

// 立即初始化导航模块
initNavigationModule();

/**
 * 导出给全局使用
 */
window.RL4CONav = {
    setActiveNavLink,
    handleHashChange,
    initSmoothScroll,
    initMobileMenu,
    showPageLoading,
    hidePageLoading,
    navigateTo,
    getCurrentPageName
};

