// 从当前标签页的提示词文本框取值，供「载入当前提示词」按钮使用。
// Forge / A1111 的惯例是直接查 DOM：gradioApp().querySelector('#txt2img_prompt textarea')
function ztpe_get_prompt(tab) {
    try {
        const el = gradioApp().querySelector('#' + tab + '_prompt textarea');
        return el ? el.value : '';
    } catch (e) {
        return '';
    }
}
